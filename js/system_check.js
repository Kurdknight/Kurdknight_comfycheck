import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// Add a button to the settings menu
app.registerExtension({
    name: "Kurdknight.SystemCheck",
    async setup() {
        // Add settings button
        const menu = document.querySelector(".comfy-menu");
        const separator = document.createElement("hr");
        const button = document.createElement("button");
        button.textContent = "System Check";
        button.onclick = () => {
            showSystemInfo();
        };
        menu.appendChild(separator);
        menu.appendChild(button);

        // Register for system check messages
        api.addEventListener("kurdknight.systemcheck.update", (evt) => {
            const data = evt.detail;
            updateSystemInfoDialog(data);
        });
    }
});

// Create a styled dialog for system information
function createStyledDialog() {
    const dialog = document.createElement("dialog");
    dialog.id = "system-check-dialog";
    dialog.style.cssText = `
        padding: 20px;
        max-width: 800px;
        max-height: 80vh;
        overflow-y: auto;
        background: #1a1a1a;
        color: #ffffff;
        border: 1px solid #333;
        border-radius: 8px;
    `;

    const style = document.createElement("style");
    style.textContent = `
        #system-check-dialog .section {
            margin: 10px 0;
            padding: 10px;
            background: #2a2a2a;
            border-radius: 4px;
        }
        #system-check-dialog .section-title {
            font-weight: bold;
            color: #00ff00;
            margin-bottom: 5px;
        }
        #system-check-dialog .warning {
            color: #ffff00;
        }
        #system-check-dialog .error {
            color: #ff0000;
        }
        #system-check-dialog .success {
            color: #00ff00;
        }
        #system-check-dialog button {
            background: #444;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            margin: 5px;
        }
        #system-check-dialog button:hover {
            background: #555;
        }
        #system-check-dialog .close-btn {
            position: absolute;
            top: 10px;
            right: 10px;
        }
        #system-check-dialog .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 5px;
        }
        #system-check-dialog .status-good {
            background-color: #00ff00;
        }
        #system-check-dialog .status-warning {
            background-color: #ffff00;
        }
        #system-check-dialog .status-error {
            background-color: #ff0000;
        }
    `;

    document.head.appendChild(style);
    return dialog;
}

// Show system information dialog
function showSystemInfo() {
    let dialog = document.getElementById("system-check-dialog");
    if (!dialog) {
        dialog = createStyledDialog();
        document.body.appendChild(dialog);
        
        // Add close button
        const closeBtn = document.createElement("button");
        closeBtn.textContent = "×";
        closeBtn.className = "close-btn";
        closeBtn.onclick = () => dialog.close();
        dialog.appendChild(closeBtn);

        // Add action buttons
        const buttonContainer = document.createElement("div");
        buttonContainer.style.textAlign = "center";
        buttonContainer.style.marginTop = "20px";

        const refreshBtn = document.createElement("button");
        refreshBtn.textContent = "Refresh";
        refreshBtn.onclick = () => {
            api.fetchApi("/kurdknight/systemcheck/refresh");
        };

        const saveBtn = document.createElement("button");
        saveBtn.textContent = "Save Report";
        saveBtn.onclick = () => {
            api.fetchApi("/kurdknight/systemcheck/save");
        };

        buttonContainer.appendChild(refreshBtn);
        buttonContainer.appendChild(saveBtn);
        dialog.appendChild(buttonContainer);
    }

    dialog.showModal();
}

// Update the dialog with system information
function updateSystemInfoDialog(data) {
    const dialog = document.getElementById("system-check-dialog");
    if (!dialog) return;

    // Clear previous content
    const content = document.createElement("div");
    
    // Parse and format the information
    const sections = data.message.split("\n=== ");
    sections.forEach(section => {
        if (!section.trim()) return;
        
        const [title, ...lines] = section.split("\n");
        const sectionDiv = document.createElement("div");
        sectionDiv.className = "section";
        
        const titleDiv = document.createElement("div");
        titleDiv.className = "section-title";
        titleDiv.textContent = title.replace("===", "").trim();
        
        const contentDiv = document.createElement("div");
        lines.forEach(line => {
            if (!line.trim()) return;
            const lineDiv = document.createElement("div");
            
            // Add status indicators
            const status = getStatusIndicator(line);
            if (status) {
                const indicator = document.createElement("span");
                indicator.className = `status-indicator status-${status}`;
                lineDiv.appendChild(indicator);
            }
            
            lineDiv.appendChild(document.createTextNode(line));
            contentDiv.appendChild(lineDiv);
        });
        
        sectionDiv.appendChild(titleDiv);
        sectionDiv.appendChild(contentDiv);
        content.appendChild(sectionDiv);
    });

    // Replace dialog content while preserving buttons
    const buttons = dialog.querySelector("div:last-child");
    dialog.innerHTML = "";
    dialog.appendChild(content);
    dialog.appendChild(buttons);
}

// Helper function to determine status indicators
function getStatusIndicator(line) {
    const lowercaseLine = line.toLowerCase();
    if (lowercaseLine.includes("error") || lowercaseLine.includes("not found") || lowercaseLine.includes("not available")) {
        return "error";
    }
    if (lowercaseLine.includes("warning")) {
        return "warning";
    }
    if (lowercaseLine.includes("available") || lowercaseLine.includes("enabled") || lowercaseLine.includes("true")) {
        return "good";
    }
    return null;
} 