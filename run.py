# flask_app.py
import os
from flask import Flask, request, render_template_string, jsonify
from app_v3 import process_url  # Import the function from your Gradio file module

# Create the Flask app instance
app = Flask(__name__)

# HTML template for the home page
HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>News Semantic Analysis & Summary</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 2em; }
      .container { max-width: 800px; margin: auto; }
      input[type="text"] { width: 100%; padding: 10px; font-size: 1em; }
      input[type="submit"] { padding: 10px 20px; font-size: 1em; }
      .result { margin-top: 20px; white-space: pre-wrap; background: #f4f4f4; padding: 15px; border-radius: 5px; }
      .error { color: red; }
    </style>
  </head>
  <body>
    <div class="container">
      <h1>News Semantic Analysis & Enhanced Summary</h1>
      <p>Enter a valid news URL to analyze its content and generate a summary.</p>
      <form method="post">
        <input type="text" name="url" placeholder="Enter news URL here..." required>
        <br><br>
        <input type="submit" value="Analyze">
      </form>
      {% if summary %}
      <div class="result">
        <h2>Summary:</h2>
        <p>{{ summary }}</p>
      </div>
      {% endif %}
      {% if error %}
      <div class="error">
        <h2>Error:</h2>
        <p>{{ error }}</p>
      </div>
      {% endif %}
    </div>
  </body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    summary = None
    error = None
    if request.method == "POST":
        url = request.form.get("url", "").strip()
        if not url:
            error = "Please provide a valid URL."
        else:
            result = process_url(url)
            # Check if the result indicates an error
            if result.startswith("⚠️"):
                error = result
            else:
                summary = result
    return render_template_string(HTML_TEMPLATE, summary=summary, error=error)

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """
    API endpoint to analyze a URL.
    Expects a JSON payload with the key 'url'.
    Returns a JSON response with the summary or error message.
    """
    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"error": "Missing 'url' in request body"}), 400
    
    url = data["url"].strip()
    if not url:
        return jsonify({"error": "URL cannot be empty"}), 400

    result = process_url(url)
    if result.startswith("⚠️"):
        return jsonify({"error": result}), 500
    return jsonify({"summary": result})

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
