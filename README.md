**README**

GPA-Qwen is an advanced AI assistant designed to help users with a wide range of tasks, including:

- Answering questions.
- Performing calculations.
- Generating content.
- Conducting research.

**Features:**

1.Multi-Tool Support
GPA-Qwen utilizes a variety of tools to enhance its functionality. These include:

- Search: For finding information online.
- Wiki: For accessing comprehensive articles from Wikipedia.
- Save: For saving output or data.
- Calculator: For performing mathematical calculations.
- Content Generator: For creating content such as text, poems, and more.
- 
2. Customizable Output:
  
Users can control which tools are used by GPA-Qwen through a checkbox interface. This allows for flexible task execution based on individual needs.

3. User-Friendly Interface:

The application is built using Gradio, providing an intuitive and easy-to-use web interface. Users can input their queries directly into the text box and select the tools they want to use. The results and intermediate steps are displayed in real-time.


**Setup:**

**Prerquisite:s**

- Python 3.8 or higher
- Required libraries (found in requirements.txt)

**Installation:**

1. Clone the repository
- 		git clone https://github.com/your-repo/gpa-qwen.git
2. Navigate to the project directory:
- 		cd gpa-qwen
3. Create a virtual enviroment (if not done so)

Windows:
- 		python -m venv venv
- 		.\venv\Scripts\activate

Linux/MacOS:
- 		python3 -m venv venv
- 		source /venv/bin/activate/

4. Install dependencies:
- 		pip install -r requirements.txt
5. Set up enviroment variables (if needed)
- 		cp .env.example .env

To run it simply run:

Windows:

		python app.py

Linux/MacOS:

		python3 app.py

**Usage**

1. Open your web browser and navigate to http://localhost:7860.
2. Enter your query in the text box at the top of the page.
3. Select any tools you want GPA-Qwen to use from the checkboxes below.
4. Click the "Submit" button to execute the task.
5. The results will be displayed in the "Summary" output box, along with intermediate steps and any applicable sources.

**Contributing**

Contributions are welcome! If you have any ideas for new features, bug fixes, or improvements, feel free to open an issue or submit a pull request.

**License**

This project is licensed under the CC0-1.0 License - see the LICENSE file for details.
