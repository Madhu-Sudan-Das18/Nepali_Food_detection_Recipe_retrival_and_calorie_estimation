import os
import torch
import json
import sqlite3
import timm
from PIL import Image
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from torchvision import transforms
import torch.nn.functional as F
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['SECRET_KEY'] = 'your-secret-key-here'  
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


MODEL_PATH = "best_food_model.pth"
RECIPES_JSON = "recipes.json"
DB_PATH = "food_history.db"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(MODEL_PATH, map_location=device)
classes = ckpt['classes']

model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=len(classes))
model.load_state_dict(ckpt['model_state'])
model.to(device)
model.eval()


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            predicted_food TEXT,
            confidence REAL,
            calories REAL,
            datetime TEXT,
            user_id INTEGER
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def init_default_user():
    """Create a default user for testing"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        
        c.execute("SELECT COUNT(*) FROM users")
        if c.fetchone()[0] == 0:
            
            password_hash = generate_password_hash("password123")
            c.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                ("demo", "demo@example.com", password_hash)
            )
            conn.commit()
            print("Default user created: username='demo', password='password123'")
    except Exception as e:
        print(f"Error creating default user: {e}")
    finally:
        conn.close()

def get_categories_for_food(food_name, recipes):
    """Extract available categories for a food item"""
    if food_name not in recipes:
        return None
    
    dish_info = recipes[food_name]
    categories = {}
    
    if "categories" in dish_info:
        for cat_name, cat_data in dish_info["categories"].items():
            
            if isinstance(cat_data, dict) and "subcategories" in cat_data:
                for subcat_name, subcat_data in cat_data["subcategories"].items():
                    if isinstance(subcat_data, dict):
                        category_key = f"{cat_name}_{subcat_name}"
                        categories[category_key] = {
                            "display_name": subcat_data.get("display_name", f"{cat_name} {subcat_name.title()}"),
                            "data": subcat_data
                        }
            
            elif isinstance(cat_data, dict) and any(key in cat_data for key in ['ingredients', 'recipe', 'base_servings']):
                categories[cat_name] = {
                    "display_name": cat_data.get("display_name", cat_name),
                    "data": cat_data
                }
    
    return categories if categories else None

def get_recipe_data(food_name, category_key, recipes):
    """Get recipe data for a specific food and category - SIMPLIFIED VERSION"""
    if food_name not in recipes:
        return None
    
    dish_info = recipes[food_name]
    
    
    if not category_key:
        return dish_info
    
    
    if "categories" in dish_info:
        
        if "_" in category_key:
            main_cat, sub_cat = category_key.split("_", 1)
            
            if (main_cat in dish_info["categories"] and 
                "subcategories" in dish_info["categories"][main_cat] and
                sub_cat in dish_info["categories"][main_cat]["subcategories"]):
                
                return dish_info["categories"][main_cat]["subcategories"][sub_cat]
        else:
            
            if category_key in dish_info["categories"]:
                cat_data = dish_info["categories"][category_key]
                
                
                if any(key in cat_data for key in ['ingredients', 'recipe', 'base_servings']):
                    return cat_data
                
                elif "subcategories" in cat_data and cat_data["subcategories"]:
                    first_subcat = next(iter(cat_data["subcategories"].values()))
                    return first_subcat
    
    
    return dish_info

def calculate_scaled_ingredients(info, num_people):
    """Calculate ingredients scaled for number of servings - SIMPLIFIED VERSION"""
    if not info or 'ingredients' not in info:
        return [], 0.0
        
    base_servings = info.get('base_servings', 1)
    scale = num_people / base_servings
    total_kcal = 0.0
    scaled_ingredients = []
    
    for ing in info['ingredients']:
        grams_per_serving = ing.get('grams_per_serving', ing.get('grams', 0))
        kcal_per_100g = ing.get('kcal_per_100g', 0)
        name = ing.get('name', 'Unknown ingredient')
        
        grams = grams_per_serving * scale
        kcal = grams * kcal_per_100g / 100.0
        total_kcal += kcal
        scaled_ingredients.append({
            "name": name,
            "grams": round(grams, 2),
            "kcal": round(kcal, 2)
        })
    
    return scaled_ingredients, round(total_kcal, 2)


try:
    with open(RECIPES_JSON, "r", encoding="utf-8") as f:
        recipes_data = json.load(f)
    print(f"DEBUG: Successfully loaded recipes. Total foods: {len(recipes_data)}")
except Exception as e:
    print(f"ERROR: Failed to load recipes: {e}")
    recipes_data = {}


@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = c.fetchone()
        conn.close()
        
        if user and check_password_hash(user[3], password):
            session['user_id'] = user[0]
            session['username'] = user[1]
            return jsonify({'success': True, 'message': 'Login successful!'})
        else:
            return jsonify({'success': False, 'message': 'Invalid username or password'})
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if not username or not email or not password:
            return jsonify({'success': False, 'message': 'All fields are required'})
        
        password_hash = generate_password_hash(password)
        
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        try:
            c.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                (username, email, password_hash)
            )
            conn.commit()
            
            
            c.execute("SELECT id FROM users WHERE username = ?", (username,))
            user_id = c.fetchone()[0]
            conn.close()
            
            session['user_id'] = user_id
            session['username'] = username
            return jsonify({'success': True, 'message': 'Registration successful!'})
            
        except sqlite3.IntegrityError:
            conn.close()
            return jsonify({'success': False, 'message': 'Username or email already exists'})
        except Exception as e:
            conn.close()
            return jsonify({'success': False, 'message': 'Registration failed'})
    
    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', username=session.get('username'))

@app.route('/health-info')
def health_info():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('health_info.html', username=session.get('username'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Authentication required'}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        file = request.files['file']
        
        
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        
        img = Image.open(filepath).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(x)
            probs = F.softmax(out, dim=1)[0]
            top3_prob, top3_idx = torch.topk(probs, 3)

        top_pred_idx = int(top3_idx[0].item())
        pred_class = classes[top_pred_idx]
        confidence = float(top3_prob[0].item() * 100)

        print(f"DEBUG: Predicted food: {pred_class} with confidence: {confidence}%")

        if pred_class not in recipes_data:
            return jsonify({"error": f"No recipe found for {pred_class}"}), 404

        
        categories = get_categories_for_food(pred_class, recipes_data)
        
        response_data = {
            "predicted_food": pred_class,
            "confidence": f"{confidence:.2f}",
            "top3": [
                {"name": classes[top3_idx[i]], "prob": f"{top3_prob[i].item()*100:.2f}"}
                for i in range(3)
            ],
            "image_url": f"/static/uploads/{filename}",
            "needs_category_selection": categories is not None and len(categories) > 0
        }
        
        if categories and len(categories) > 0:
            response_data["categories"] = categories
        else:
            
            dish_info = recipes_data[pred_class]
            
            
            if "categories" in dish_info:
                
                first_cat = next(iter(dish_info["categories"].values()))
                info = first_cat
            else:
                info = dish_info
            
            scaled_ingredients, total_kcal = calculate_scaled_ingredients(info, 1)
            recipe_steps = info.get("recipe", "Recipe not available")
            display_name = info.get("display_name", pred_class)
            
            response_data.update({
                "display_name": display_name,
                "ingredients": scaled_ingredients,
                "calories": total_kcal,
                "recipe": recipe_steps
            })
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"ERROR in predict: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_recipe', methods=['POST'])
@login_required
def get_recipe():
    try:
        data = request.json
        food_name = data.get('food_name')
        category_key = data.get('category_key')
        num_people = int(data.get('servings', 1))
        filename = data.get('filename')

        print(f"DEBUG: get_recipe called with food={food_name}, category={category_key}, servings={num_people}")

        if not food_name:
            return jsonify({"error": "Food name is required"}), 400

        if food_name not in recipes_data:
            return jsonify({"error": f"Food '{food_name}' not found in recipes database"}), 404

        dish_info = recipes_data[food_name]
        
        
        if "categories" in dish_info and category_key and category_key in dish_info["categories"]:
            info = dish_info["categories"][category_key]
        elif "categories" in dish_info:
            
            first_cat = next(iter(dish_info["categories"].values()))
            info = first_cat
        else:
            info = dish_info

        
        scaled_ingredients, total_kcal = calculate_scaled_ingredients(info, num_people)
        
        if not scaled_ingredients:
            return jsonify({"error": "No ingredients data found for this food item."}), 404

        recipe_steps = info.get("recipe", "Recipe instructions not available")
        display_name = info.get("display_name", food_name)

        
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO history (filename, predicted_food, confidence, calories, datetime, user_id) VALUES (?, ?, ?, ?, ?, ?)",
                  (filename, f"{food_name} ({display_name})", 0, total_kcal, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), session['user_id']))
        conn.commit()
        conn.close()

        return jsonify({
            "display_name": display_name,
            "ingredients": scaled_ingredients,
            "calories": total_kcal,
            "recipe": recipe_steps,
            "base_servings": info.get('base_servings', 1)
        })
        
    except Exception as e:
        print(f"ERROR in get_recipe: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/history', methods=['GET'])
@login_required
def get_history():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM history WHERE user_id = ? ORDER BY datetime DESC", (session['user_id'],))
    rows = c.fetchall()
    conn.close()

    history = []
    for r in rows:
        history.append({
            "filename": r[1],
            "predicted_food": r[2],
            "confidence": r[3],
            "calories": r[4],
            "datetime": r[5],
        })
    return jsonify(history)


if __name__ == '__main__':
    init_db()
    init_default_user()
    app.run(debug=True, port=5000)