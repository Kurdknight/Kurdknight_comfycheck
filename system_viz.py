import os
import json
from datetime import datetime

class SystemVizNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "system_info": ("SYSTEM_INFO",),
        }}
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "create_viz"
    CATEGORY = "utils"

    def __init__(self):
        self.output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                     "output", "system_reports")
        os.makedirs(self.output_dir, exist_ok=True)

    def create_html_report(self, data):
        """Create an HTML report with tabs and visualizations"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"system_report_{timestamp}.html"
        filepath = os.path.join(self.output_dir, filename)

        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>System Information Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #1a1a1a; color: #fff; }
                .tab-container { margin-top: 20px; }
                .tab-buttons { overflow: hidden; border: 1px solid #333; background: #2a2a2a; }
                .tab-buttons button {
                    background: inherit;
                    border: none;
                    outline: none;
                    cursor: pointer;
                    padding: 14px 16px;
                    color: #fff;
                }
                .tab-buttons button:hover { background: #3a3a3a; }
                .tab-buttons button.active { background: #4a4a4a; }
                .tab-content {
                    display: none;
                    padding: 20px;
                    border: 1px solid #333;
                    border-top: none;
                    background: #2a2a2a;
                }
                .info-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    margin-top: 20px;
                }
                .info-card {
                    background: #3a3a3a;
                    padding: 15px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                }
                .info-card h3 { margin-top: 0; color: #00ff00; }
                .status-good { color: #00ff00; }
                .status-warning { color: #ffff00; }
                .status-error { color: #ff0000; }
                .meter {
                    height: 20px;
                    background: #444;
                    border-radius: 10px;
                    padding: 2px;
                    margin-top: 5px;
                }
                .meter > span {
                    display: block;
                    height: 100%;
                    border-radius: 8px;
                    background-color: #00ff00;
                    position: relative;
                    overflow: hidden;
                }
                .copy-button {
                    float: right;
                    padding: 5px 10px;
                    background: #444;
                    border: none;
                    color: white;
                    border-radius: 4px;
                    cursor: pointer;
                }
                .copy-button:hover { background: #555; }
            </style>
        </head>
        <body>
            <h1>System Information Report</h1>
            <div class="tab-container">
                <div class="tab-buttons">
        """

        # Add tab buttons
        for section in data.keys():
            section_id = section.lower().replace(" ", "-")
            html += f'<button class="tab-button" onclick="openTab(event, \'{section_id}\')">{section}</button>'

        html += """
                </div>
        """

        # Add tab contents
        for section, section_data in data.items():
            section_id = section.lower().replace(" ", "-")
            html += f"""
                <div id="{section_id}" class="tab-content">
                    <h2>{section}</h2>
                    <button class="copy-button" onclick="copySection('{section_id}')">Copy Section</button>
                    <div class="info-grid">
            """

            # Create cards for each piece of information
            for key, value in section_data.items():
                status_class = ""
                if isinstance(value, str):
                    if "true" in value.lower() or "available" in value.lower():
                        status_class = "status-good"
                    elif "false" in value.lower() or "not found" in value.lower():
                        status_class = "status-error"

                html += f"""
                        <div class="info-card">
                            <h3>{key}</h3>
                            <p class="{status_class}">{value}</p>
                """

                # Add visualization for percentage values
                if isinstance(value, str) and "%" in value:
                    try:
                        percentage = float(value.strip("%"))
                        html += f"""
                            <div class="meter">
                                <span style="width: {percentage}%"></span>
                            </div>
                        """
                    except ValueError:
                        pass

                html += "</div>"

            html += """
                    </div>
                </div>
            """

        html += """
            </div>
            <script>
                function openTab(evt, tabName) {
                    var i, tabcontent, tabbuttons;
                    tabcontent = document.getElementsByClassName("tab-content");
                    for (i = 0; i < tabcontent.length; i++) {
                        tabcontent[i].style.display = "none";
                    }
                    tabbuttons = document.getElementsByClassName("tab-button");
                    for (i = 0; i < tabbuttons.length; i++) {
                        tabbuttons[i].className = tabbuttons[i].className.replace(" active", "");
                    }
                    document.getElementById(tabName).style.display = "block";
                    evt.currentTarget.className += " active";
                }

                function copySection(sectionId) {
                    const section = document.getElementById(sectionId);
                    const text = section.innerText;
                    navigator.clipboard.writeText(text);
                }

                // Open first tab by default
                document.getElementsByClassName("tab-button")[0].click();
            </script>
        </body>
        </html>
        """

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)

        return filepath

    def create_viz(self, system_info):
        # Create HTML report
        html_path = self.create_html_report(system_info)
        return (f"HTML report saved to: {html_path}",) 