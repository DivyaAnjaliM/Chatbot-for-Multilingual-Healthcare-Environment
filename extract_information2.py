import os

def read_reports_from_folder(folder_path):
    """
    Reads all text files from the specified folder and returns a list of report contents.
    
    Args:
        folder_path (str): The path to the folder containing medical reports.
        
    Returns:
        List[Dict[str, str]]: A list of dictionaries where each dictionary contains 'file_name' and 'content'.
    """
    reports = []
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder '{folder_path}' does not exist.")
    
    # Loop through all files in the folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        
        # Check if it's a file (not a folder) and ends with .txt (assuming reports are in text format)
        if os.path.isfile(file_path) and file_name.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
                # Append the report details (filename and content) to the list
                reports.append({
                    'file_name': file_name,
                    'content': content
                })
    
    if not reports:
        raise ValueError("No reports found in the folder.")
    
    return reports
