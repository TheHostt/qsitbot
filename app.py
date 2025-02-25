from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import json
import os
import re
import subprocess
import google.generativeai as genai
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer, util

# Configure Google Gemini API
GEMINI_API_KEY = "AIzaSyDBr0yS8eZVTVltRzR3QxhSvJd3K4D3Hs8" 

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)
CHART_FOLDER = "static/charts"
MODIFIED_FOLDER = "templates"
DATA_FOLDER = "templates/data"
UPLOAD_FOLDER = "templates/data"


for folder in [CHART_FOLDER, MODIFIED_FOLDER,UPLOAD_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)
        
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

chart_instructions = ["create chart", "generate chart", "visualize data", "biggest", "lowest", "histogram"]
map_instructions = ["modify map", "change basemap", "update map", "edit map"]
geojson_instructions = ["add point", "change point color", "change point name","change layer color", "remove point","change polygon","modify polygon"]
buffer_instructions = ["buffer", "add buffer around points", "add buffer layer", "change buffer color", "modify / change bufferdistance",]
heatmap_instructions = ["create heatmap", "add heatmap layer", "heatmap"]
closest_join_instructions = ["add closest join", "create closest join","spatial join closest"]
intersect_join_instructions = ["add intersect join", "create intersect join", "between polygons and points", "spatial join"]


genai.configure(api_key=GEMINI_API_KEY)
llm_model = genai.GenerativeModel(model_name="gemini-2.0-flash")

# chart_template = ChatPromptTemplate.from_template("""
# You are an AI that generates a complete HTML page with Chart.js based on JSON data.
# Ensure that the response contains the complete modified code, preserving all other existing code while applying the requested modification.
# Do not use placeholders or shortcuts like '...'. Return the full modified code completely like the given code with the modification.
# i want only the complete code with the new modification in the response, dont add any comments in the code
# User Instruction: {instruction}
# JSON Data: {json_data}
# Generate a full HTML file including a Chart.js script:
# """)
# chart_chain = chart_template | llm_model

# map_template = ChatPromptTemplate.from_template("""
# You are a smart AI that modifies ArcGIS JavaScript code based on user instructions.
# Ensure that the response contains the complete modified code, preserving all other existing code while applying the requested modification.
# Modify the Map HTML code accordingly.
# User Instruction: {instruction}
# Code:
# {js_code}
# Modified Code:
# """)
# map_chain = map_template | llm_model

# geojson_template = ChatPromptTemplate.from_template("""
# You are an AI assistant for modifying GeoJSON files. 
# Follow these rules:
# - Select the most appropriate GeoJSON file based on the given instruction.
# - Generate Python code that modifies the selected file based on user needs.
# - Ensure the Python code iterates over the file correctly and updates it.
# - Do not return explanations, only the complete executable Python code.

# Instruction: {instruction}
# Available Files: {file_list}
# File Structures: {file_structures}
# Python Code:
# """)

# buffer_map_template = ChatPromptTemplate.from_template("""
# You are an AI specialized in modifying ArcGIS JavaScript code based on precise user instructions. 
# if the user instruction contain modify thing like parameters add the modify if not generate for me only copy of this code "{sample_code}" in the response
# User Instruction: {instruction}
# Code:
# """)
# buffer_map_chain = buffer_map_template | llm_model

# heatmap_map_template = ChatPromptTemplate.from_template("""
# You are an AI specialized in modifying ArcGIS JavaScript code **only when explicitly instructed**. 
# Follow these rules strictly:
# - If the user does not specify a modification, return the original code **unchanged**.
# - If a modification is requested, apply **only** that modification while keeping the rest of the code identical.
# - Do not alter the logical flow, functions, or structure of the code.
# - Preserve the exact formatting, indentation, and structure of the original code.
# - Do not add, remove, or comment on any part of the code unless explicitly requested.
# - Ensure the modified code remains fully functional.

# Original Code:
# {sample_code}

# User Instruction:
# {instruction}

# Modified Code:
# """)

# heatmap_map_chain = heatmap_map_template | llm_model

# closest_join_map_template = ChatPromptTemplate.from_template("""
# You are an AI specialized in modifying ArcGIS JavaScript code **only when explicitly instructed**. 
# Follow these rules strictly:
# - If the user does not specify a modification, return the original code **unchanged**.
# - If a modification is requested, apply **only** that modification while keeping the rest of the code identical.
# - Do not alter the logical flow, functions, or structure of the code.
# - Preserve the exact formatting, indentation, and structure of the original code.
# - Do not add, remove, or comment on any part of the code unless explicitly requested.
# - Ensure the modified code remains fully functional.

# Original Code:
# {sample_code}

# User Instruction:
# {instruction}

# Modified Code:
# """)

# closest_join_map_chain = closest_join_map_template | llm_model

# intersect_join_map_template = ChatPromptTemplate.from_template("""
# You are an AI specialized in modifying ArcGIS JavaScript code **only when explicitly instructed**. 
# Follow these rules strictly:
# - If the user does not specify a modification, return the original code **unchanged**.
# - If a modification is requested, apply **only** that modification while keeping the rest of the code identical.
# - Do not alter the logical flow, functions, or structure of the code.
# - Preserve the exact formatting, indentation, and structure of the original code.
# - Do not add, remove, or comment on any part of the code unless explicitly requested.
# - Ensure the modified code remains fully functional.

# Original Code:
# {sample_code}

# User Instruction:
# {instruction}

# Modified Code:
# """)

# intersect_join_map_chain = intersect_join_map_template | llm_model

@app.route("/")
def home():
    return render_template("UI.html")

@app.route("/esri_map")
def esri_map():
    return render_template("map.html")

def extract_code(response):
    """Extracts valid Python, HTML, JavaScript, or JSON code from the LLM response while removing AI-generated explanations."""
    
    # Extract Python code
    match = re.search(r"```python\n([\s\S]*?)\n```", response, re.DOTALL)
    if match:
        extracted_code = match.group(1).strip()
        print("\n[DEBUG] Extracted Python Code:\n", extracted_code)
        return extracted_code
    
    # Extract HTML code
    match = re.search(r"(<html[\s\S]*?</html>)", response, re.DOTALL)
    if match:
        extracted_code = match.group(1).strip()
        print("\n[DEBUG] Extracted HTML Code:\n", extracted_code)
        return extracted_code
    
    # Extract JavaScript code
    match = re.search(r"(<script[\s\S]*?</script>)", response, re.DOTALL)
    if match:
        extracted_code = match.group(1).strip()
        print("\n[DEBUG] Extracted JavaScript Code:\n", extracted_code)
        return extracted_code
    
    # Extract JSON or other structured code formats
    match = re.search(r"```(?:html|javascript|json)?\s*(.*?)\s*```", response, re.DOTALL)
    extracted_code = match.group(1).strip() if match else response.strip()
    
    # Remove AI-generated explanations
    extracted_code = re.sub(r"This new script.*?$", "", extracted_code, flags=re.MULTILINE).strip()
    
    print("\n[DEBUG] Extracted Code (Fallback):\n", extracted_code)
    
    return extracted_code


@app.route("/list-files")
def list_files():
    """Returns a list of all GeoJSON files with proper paths."""
    try:
        files = [f"/data/{file}" for file in os.listdir(DATA_FOLDER) if file.endswith(".json")]
        return jsonify(files)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/data/<filename>")
def serve_json(filename):
    """Serves JSON files from the templates/data directory."""
    return send_from_directory("templates/data", filename)

@app.route("/upload", methods=["POST"])
def upload_file():
    """Handles multiple file uploads and stores them in the data folder."""
    if "files" not in request.files:
        return jsonify({"error": "No files uploaded"}), 400
    files = request.files.getlist("files")
    uploaded_files = []
    for file in files:
        if file.filename == "":
            continue
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)  
        file.save(file_path)
        uploaded_files.append({
            "file_name": file.filename,
            "file_path": file_path
        })
    return jsonify({
        "message": "Files uploaded and stored successfully!",
        "uploaded_files": uploaded_files
    })

@app.route("/process", methods=["POST"])
def process_input():
    """Determines whether to modify GeoJSON, modify a map, generate a chart, or add buffer functionality."""
    data = request.get_json()
    user_input = data.get("instruction", "")
    file_path = data.get("file_path", "")

    if not user_input:
        return jsonify({"error": "Instruction is required"}), 400

    # Compute similarity scores
    user_embedding = sbert_model.encode([user_input], convert_to_tensor=True)
    chart_score = util.pytorch_cos_sim(user_embedding, sbert_model.encode(chart_instructions, convert_to_tensor=True)).max().item()
    map_score = util.pytorch_cos_sim(user_embedding, sbert_model.encode(map_instructions, convert_to_tensor=True)).max().item()
    geojson_score = util.pytorch_cos_sim(user_embedding, sbert_model.encode(geojson_instructions, convert_to_tensor=True)).max().item()
    buffer_score = util.pytorch_cos_sim(user_embedding, sbert_model.encode(buffer_instructions, convert_to_tensor=True)).max().item()
    heatmap_score = util.pytorch_cos_sim(user_embedding, sbert_model.encode(heatmap_instructions, convert_to_tensor=True)).max().item()
    closest_join_score = util.pytorch_cos_sim(user_embedding, sbert_model.encode(closest_join_instructions, convert_to_tensor=True)).max().item()
    intersect_join_score = util.pytorch_cos_sim(user_embedding, sbert_model.encode(intersect_join_instructions, convert_to_tensor=True)).max().item()

    # Debugging: Print Scores
    print(f"Scores:\n Chart: {chart_score}\n Map: {map_score}\n GeoJSON: {geojson_score}")
    print(f" Buffer: {buffer_score}\n Heatmap: {heatmap_score}\n Closest Join: {closest_join_score}\n Intersect Join: {intersect_join_score}")

    # 🔹 Fix Prioritization Order

    # Highest priority: Ensure intersect and closest join work correctly
    if intersect_join_score > heatmap_score and intersect_join_score > closest_join_score and intersect_join_score > geojson_score:
        return modify_map_with_intersect_join(user_input)
    
    if closest_join_score > heatmap_score and closest_join_score > geojson_score:
        return modify_map_with_closest_join(user_input)

    # Ensure heatmap is handled correctly before map modifications
    if heatmap_score > buffer_score and heatmap_score > chart_score and heatmap_score > map_score:
        return modify_map_with_heatmap(user_input)

    # Buffer-based modifications
    if buffer_score > chart_score and buffer_score > map_score:
        return modify_map_with_buffer(user_input)

    # General map modifications (Basemap, layers, etc.)
    if map_score > chart_score and map_score > geojson_score:
        return modify_map(user_input)

    # Ensure geojson changes only apply when it's the most relevant
    if geojson_score > chart_score and geojson_score > buffer_score and geojson_score > map_score and geojson_score > intersect_join_score:
        return modify_geojson()

    # Chart generation
    if chart_score > map_score:
        return generate_chart(file_path, user_input)

    return jsonify({"error": "No valid process matched"}), 400


def extract_json_structure():
    """Extracts the structure of all GeoJSON files in UPLOAD_FOLDER and returns their paths."""
    file_structures = {}
    try:
        files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith(".json")]
        for file in files:
            path = os.path.join(UPLOAD_FOLDER, file)
            with open(path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    structure = {
                        "geometry": list(data["features"][0]["geometry"].keys()) if "features" in data and data["features"] else [],
                        "properties": list(data["features"][0]["properties"].keys()) if "features" in data and data["features"] else []
                    }
                    file_structures[file] = {"path": path, "structure": structure}
                except Exception as e:
                    print(f"Skipping {file} due to error: {str(e)}")
        
        print("\n[DEBUG] Extracted JSON Structure:")
        for file, info in file_structures.items():
            print(f"  - {file}: Path={info['path']}, Structure={info['structure']}")
        
        return file_structures
    except Exception as e:
        print("Error extracting JSON structures:", str(e))
        return {}

def generate_chart(file_path, instruction):
    """Generates a complete HTML file with Chart.js based on extracted JSON structure using Google Gemini API."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        # 🔹 Create a structured prompt for Gemini API
        prompt = f"""
        You are an AI that generates a complete HTML page with Chart.js based on JSON data.
        Ensure that the response contains the complete modified code, preserving all other existing code while applying the requested modification.
        Do not use placeholders or shortcuts like '...'. Return the full modified code completely like the given code with the modification.
        I want only the complete code with the new modification in the response, don't add any comments in the code.

        User Instruction: {instruction}
        JSON Data: {json.dumps(json_data)}

        Generate a full HTML file including a Chart.js script:
        """

        # 🔹 Call Google Gemini API
        response = llm_model.generate_content(prompt).text.strip()


        # 🔹 Extract HTML Code
        html_code = extract_code(response)

        if not html_code:
            return jsonify({"error": "Failed to extract valid HTML"}), 500

        # Save the chart HTML file
        chart_file_path = os.path.join(CHART_FOLDER, "generated_chart.html")
        with open(chart_file_path, "w", encoding="utf-8") as f:
            f.write(html_code)

        return jsonify({
            "message": "Chart HTML generated successfully!",
            "chart_file_path": chart_file_path
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/modify-geojson", methods=["POST"])
def modify_geojson():
    """Determines the correct GeoJSON file, generates a modification script, and applies the changes."""
    data = request.get_json()
    instruction = data.get("instruction", "")

    if not instruction:
        print("[ERROR] No instruction provided.")
        return jsonify({"error": "Instruction is required"}), 400

    # Step 1: Extract JSON file structures
    json_files = extract_json_structure()

    if not json_files:
        print("[ERROR] No valid GeoJSON files found.")
        return jsonify({"error": "No valid GeoJSON files found"}), 400

    # Step 2: Format file data for LLM
    file_list = [f"{name} (Path: {info['path']})" for name, info in json_files.items()]
    file_structures = {name: info["structure"] for name, info in json_files.items()}

    print(f"\n[DEBUG] User Instruction: {instruction}")
    print(f"[DEBUG] Available Files: {file_list}")

    # Step 3: Create a structured prompt for Gemini API
    prompt = f"""
    You are an AI assistant for modifying GeoJSON files. 
    Follow these rules:
    - Identify the correct JSON file to modify based on the user instruction.
    - Do NOT create a new file; modify the existing file in place.
    - Always read and write JSON files using UTF-8 encoding.
    - If the file has a 'features' key, iterate through its list and update the relevant field.
    - If 'features' does not exist, modify the property directly in the JSON structure.
    - Ensure the script prints debug outputs before and after changes.
    - Do not return explanations, only the complete executable Python code.

    Instruction: {instruction}
    Available Files: {json.dumps(file_list)}
    File Structures: {json.dumps(file_structures)}

    Python Code:
    """

    # Step 4: Call Google Gemini API
    response = llm_model.generate_content(prompt).text.strip()


    # Step 5: Extract Python Code from LLM Response
    extracted_code = extract_code(response)

    if not extracted_code.strip():
        print("[ERROR] Failed to generate valid modification script.")
        return jsonify({"error": "Failed to generate valid modification script."}), 500

    print("\n[DEBUG] Generated Python Code:\n", extracted_code)

    # Step 6: Execute the generated Python script safely
    try:
        exec(extracted_code)
        print("[SUCCESS] GeoJSON modification applied successfully.")
        return jsonify({"message": "GeoJSON modified successfully!", "script": extracted_code})
    except KeyError as e:
        print(f"[ERROR] KeyError: {str(e)} - Check if the JSON structure is correct.")
        return jsonify({"error": f"KeyError: {str(e)}"}), 500
    except Exception as e:
        print(f"[ERROR] Failed to execute modification script: {str(e)}")
        return jsonify({"error": str(e)}), 500



def modify_map(instruction):
    """Modifies an existing Map (ArcGIS JavaScript) based on user input using Google Gemini API."""
    try:
        # Load the existing map.html from the MODIFIED_FOLDER
        map_file_path = os.path.join(MODIFIED_FOLDER, "map.html")
        with open(map_file_path, "r", encoding="utf-8") as f:
            js_code = f.read()

        # 🔹 Create a structured prompt for Gemini API
        prompt = f"""
        You are an AI that modifies ArcGIS JavaScript code based on user instructions.
        - Apply only the modifications requested by the user.
        - Do not alter the logical flow or functionality of the code.
        - Do not add, remove, or comment on any part of the code unless instructed.
        - Maintain the exact spacing, indentation, and formatting.

        Original Code:
        {js_code}

        User Instruction:
        {instruction}

        Modified Code:
        """

        # 🔹 Call Google Gemini API
        response = llm_model.generate_content(prompt).text.strip()


        # 🔹 Debugging: Print raw response
        print(f"Raw LLM Response:\n{response}\n")

        # 🔹 Extract clean JavaScript/HTML code
        modified_js_code = extract_code(response)

        # 🔹 Debugging: Print extracted code
        print(f"Extracted Code:\n{modified_js_code}\n")

        # Ensure the extracted code is valid
        if not modified_js_code.strip():
            return jsonify({"error": "LLM response was empty or invalid."}), 500

        # Save the modified file to modified/map.html
        modified_map_path = os.path.join(MODIFIED_FOLDER, "map.html")
        with open(modified_map_path, "w", encoding="utf-8") as f:
            f.write(modified_js_code)

        return jsonify({
            "message": "Map modified successfully!",
            "map_file_path": modified_map_path
        })

    except Exception as e:
        print(f" Error in modify_map: {e}")  # Print error for debugging
        return jsonify({"error": str(e)}), 500


def modify_map_with_buffer(instruction):
    """Modifies map.html to add buffer logic using ArcGIS JS, JSON file structure, and Google Gemini API."""
    try:
        # Load existing map.html file
        map_file_path = os.path.join(MODIFIED_FOLDER, "map.html")
        with open(map_file_path, "r", encoding="utf-8") as f:
            js_code = f.read()

        # Load sample map buffer file
        sample_file_path = "sample_map_buffer.html"
        with open(sample_file_path, "r", encoding="utf-8") as f:
            sample_code = f.read()

        # Extract structure of JSON files (GeoJSON data)
        json_files = extract_json_structure()

        if not json_files:
            return jsonify({"error": "No valid GeoJSON files found"}), 400

        # Format JSON structure for Gemini
        file_list = [f"{name} (Path: {info['path']})" for name, info in json_files.items()]
        file_structures = {name: info["structure"] for name, info in json_files.items()}

        # 🔹 Enhanced Prompt: Provide JSON file structure
        prompt = f"""
        You are an AI specialized in modifying ArcGIS JavaScript code to add buffer functionality based on user instructions.

        - **Use the provided JSON files** to understand the available geographic data.
        - Apply only the modifications requested by the user.
        - Do NOT alter the logical flow or functionality of the existing code.
        - Ensure that the buffer logic is correctly applied based on the JSON file structures.
        - Maintain the exact spacing, indentation, and formatting.
        - if the User instruction mention place like egypt, UEA, KSA, USA, france, ....
            make the extend of map on this place

        ## User Instruction:
        {instruction}

        ## Available JSON Files:
        {json.dumps(file_list)}

        ## JSON File Structures:
        {json.dumps(file_structures)}

        ## Sample Code for Buffer Logic:
        {sample_code}

        ## Modified Full HTML Code (including buffer logic using JSON data):
        """

        # 🔹 Call Google Gemini API
        response = llm_model.generate_content(prompt).text.strip()

        # 🔹 Debugging: Print raw response
        print(f"Raw LLM Response:\n{response}\n")

        # 🔹 Extract clean JavaScript/HTML code
        modified_js_code = extract_code(response)

        # 🔹 Debugging: Print extracted code
        print(f"Extracted Code:\n{modified_js_code}\n")

        # Ensure the extracted code is valid
        if not modified_js_code.strip():
            return jsonify({"error": "LLM response was empty or invalid."}), 500

        # Save the modified file to modified/map.html
        with open(map_file_path, "w", encoding="utf-8") as f:
            f.write(modified_js_code)

        return jsonify({
            "message": "Map modified successfully with buffer logic!",
            "buffer_map_file_path": map_file_path
        })

    except Exception as e:
        print(f" Error in modify_map_with_buffer: {e}")  
        return jsonify({"error": str(e)}), 500

@app.route("/modify-map-buffer", methods=["POST"])
def modify_map_buffer():
    """API Endpoint to modify map.html and add buffer functionality."""
    data = request.get_json()
    user_instruction = data.get("instruction", "")
    if not user_instruction:
        return jsonify({"error": "Instruction is required"}), 400
    return modify_map_with_buffer(user_instruction)

def modify_map_with_heatmap(instruction):
    """Modifies map.html to add heatmap functionality using ArcGIS JS and Google Gemini API."""
    try:
        # Load existing map.html file
        map_file_path = os.path.join(MODIFIED_FOLDER, "map.html")
        

        # Load sample heatmap logic file
        sample_file_path = "heat_map.html"
        with open(sample_file_path, "r", encoding="utf-8") as f:
            sample_code = f.read()

        # 🔹 Create a structured prompt for Gemini API
        prompt = f"""
        You are an AI specialized in modifying ArcGIS JavaScript code to add heatmap functionality based on user instructions.
        - Apply only the modifications requested by the user.
        - Do not alter the logical flow or functionality of the code.
        - Do not add, remove, or comment on any part of the code unless instructed.
        - Maintain the exact spacing, indentation, and formatting.

        Sample Code for Heatmap Logic:
        {sample_code}

        User Instruction:
        {instruction}

        Modified Code:
        """

        # 🔹 Call Google Gemini API
        response = llm_model.generate_content(prompt).text.strip()


        # 🔹 Debugging: Print raw response
        print(f"Raw LLM Response:\n{response}\n")

        # 🔹 Extract clean JavaScript/HTML code
        modified_js_code = extract_code(response)

        # 🔹 Debugging: Print extracted code
        print(f"Extracted Code:\n{modified_js_code}\n")

        # Ensure the extracted code is valid
        if not modified_js_code.strip():
            return jsonify({"error": "LLM response was empty or invalid."}), 500

        # Save the modified file to modified/map.html
        with open(map_file_path, "w", encoding="utf-8") as f:
            f.write(modified_js_code)

        return jsonify({
            "message": "Map modified successfully with heatmap layer!",
            "heatmap_map_file_path": map_file_path
        })

    except Exception as e:
        print(f" Error in modify_map_with_heatmap: {e}")  
        return jsonify({"error": str(e)}), 500


@app.route("/modify-map-heatmap", methods=["POST"])
def modify_map_heatmap():
    """API Endpoint to modify map.html and add heatmap functionality."""
    data = request.get_json()
    user_instruction = data.get("instruction", "")
    if not user_instruction:
        return jsonify({"error": "Instruction is required"}), 400
    return modify_map_with_heatmap(user_instruction)


def modify_map_with_closest_join(instruction):
    """Modifies map.html to add closest join functionality using ArcGIS JS and Google Gemini API."""
    try:
        # Load existing map.html file
        map_file_path = os.path.join(MODIFIED_FOLDER, "map.html")
        with open(map_file_path, "r", encoding="utf-8") as f:
            js_code = f.read()

        # Load sample closest join logic file
        sample_file_path = "colsest_join.html"
        with open(sample_file_path, "r", encoding="utf-8") as f:
            sample_code = f.read()

        # 🔹 Create a structured prompt for Gemini API
        prompt = f"""
        You are an AI specialized in modifying ArcGIS JavaScript code to add closest join functionality based on user instructions.
        - Apply only the modifications requested by the user.
        - Do not alter the logical flow or functionality of the code.
        - Do not add, remove, or comment on any part of the code unless instructed.
        - Maintain the exact spacing, indentation, and formatting.

        Sample Code for Closest Join Logic:
        {sample_code}

        User Instruction:
        {instruction}

        Modified Code:
        """

        # 🔹 Call Google Gemini API
        response = llm_model.generate_content(prompt).text.strip()

        # 🔹 Debugging: Print raw response
        print(f"Raw LLM Response:\n{response}\n")

        # 🔹 Extract clean JavaScript/HTML code
        modified_js_code = extract_code(response)

        # 🔹 Debugging: Print extracted code
        print(f"Extracted Code:\n{modified_js_code}\n")

        # Ensure the extracted code is valid
        if not modified_js_code.strip():
            return jsonify({"error": "LLM response was empty or invalid."}), 500

        # Save the modified file to modified/map.html
        with open(map_file_path, "w", encoding="utf-8") as f:
            f.write(modified_js_code)

        return jsonify({
            "message": "Map modified successfully with closest join functionality!",
            "closest_join_map_file_path": map_file_path
        })

    except Exception as e:
        print(f" Error in modify_map_with_closest_join: {e}")  
        return jsonify({"error": str(e)}), 500


@app.route("/modify-map-closest-join", methods=["POST"])
def modify_map_closest_join():
    """API Endpoint to modify map.html and add closest join functionality."""
    data = request.get_json()
    user_instruction = data.get("instruction", "")
    if not user_instruction:
        return jsonify({"error": "Instruction is required"}), 400
    return modify_map_with_closest_join(user_instruction)


def modify_map_with_intersect_join(instruction):
    """Modifies map.html to add intersect join functionality using ArcGIS JS and Google Gemini API."""
    try:
        # Load existing map.html file
        map_file_path = os.path.join(MODIFIED_FOLDER, "map.html")
        with open(map_file_path, "r", encoding="utf-8") as f:
            js_code = f.read()

        # Load sample intersect join logic file
        sample_file_path = "intersect_join.html"
        with open(sample_file_path, "r", encoding="utf-8") as f:
            sample_code = f.read()

        # ✅ Extract JSON structures (RAW)
        json_files = extract_json_structure()

        # ✅ Clean JSON paths from "templates/data/{file}" → "/data/{file}"
        cleaned_json_files = {
            name: {
                "path": f"/data/{name}",  # ✅ Corrected path
                "structure": info["structure"]
            } for name, info in json_files.items()
        }

        # 🔹 Improved Prompt: Ensure full HTML structure
        prompt = f"""
        You are an AI that modifies ArcGIS JavaScript code to **add Intersect Join functionality** based on user instructions.

        - **IMPORTANT: You must return a complete HTML file, not just JavaScript.**
        - Integrate the intersect join logic inside the correct `<script>` section.
        - Do NOT replace existing JavaScript unless necessary.
        - Ensure the new logic properly integrates with the map and layers.
        - If `polygonLayer` and `pointLayer` do not exist, define them using `GeoJSONLayer`.
        - The JSON data files available are: {json.dumps(cleaned_json_files, indent=2)}

        ## User Instruction:
        {instruction}

        ## Sample Code for Intersect Join Logic:
        {sample_code}

        ## Modified Full HTML Code:
        """

        # 🔹 Call Google Gemini API
        response = llm_model.generate_content(prompt).text.strip()

        # 🔹 Debugging: Print raw response
        print(f"Raw LLM Response:\n{response}\n")

        # 🔹 Extract clean JavaScript/HTML code
        modified_html_code = extract_code(response)

        # 🔹 Debugging: Print extracted code
        print(f"Extracted Code:\n{modified_html_code}\n")

        # Ensure the extracted code is valid
        if not modified_html_code.strip():
            return jsonify({"error": "LLM response was empty or invalid."}), 500

        # Save the modified file to modified/map.html
        with open(map_file_path, "w", encoding="utf-8") as f:
            f.write(modified_html_code)

        return jsonify({
            "message": "Map modified successfully with intersect join functionality!",
            "intersect_join_map_file_path": map_file_path
        })

    except Exception as e:
        print(f" Error in modify_map_with_intersect_join: {e}")  
        return jsonify({"error": str(e)}), 500


@app.route("/modify-map-intersect-join", methods=["POST"])
def modify_map_intersect_join():
    """API Endpoint to modify map.html and add intersect join functionality."""
    data = request.get_json()
    user_instruction = data.get("instruction", "")
    if not user_instruction:
        return jsonify({"error": "Instruction is required"}), 400
    return modify_map_with_intersect_join(user_instruction)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
