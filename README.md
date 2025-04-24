<h1><b></b>GenomeLREP</b></h1>

<h2><b>Description:</b></h2>

GenomeLRep is a web-based platform designed to process and analyze genomic datasets through advanced regularization techniques. The project involved developing an interactive website that allows users to upload and work with complex genomic data efficiently.

We implemented various regularization algorithms such as Lasso, Ridge Regression, Elastic Net, and Principal Component Analysis (PCA) to clean, compress, and prepare the data for deeper statistical and machine learning analysis. These techniques helped to reduce overfitting, manage multicollinearity, and extract meaningful patterns from high-dimensional genetic data.

Once the data was regularized, it was further utilized for regression modeling to identify relationships between genetic variables and target outcomes. Additionally, the platform includes features for graphical visualization, enabling users to explore data trends and model results through dynamic plots and charts.

The goal of GenomeLRep is to simplify the process of genomic data preparation and modeling, making powerful bioinformatics tools more accessible to researchers, data scientists, and healthcare professionals.

<h2>🚀 Getting Started</h2>
<b>Most important thing to note is that this project was bulit by our team from scratch and also tested in Pycharm(Educational License).</b>

Follow these steps to set up and run the project in an isolated Python virtual environment:

<h3>1. Clone the Repository</h3>
Open your terminal or command prompt and run:

```
git clone https://github.com/Umamaheshwarmydam/GenomeLREP1.git
cd GenomeLREP1
```

<h3>2. Set Up a Virtual Environment</h3>
Creating a virtual environment ensures that project dependencies are isolated from your global Python installation.​
<li>On Windows:</li>

```
python -m venv venv
venv\Scripts\activate
```

<li>On macOS/Linux:</li>

```
python3 -m venv venv
source venv/bin/activate
```

Once activated, your terminal prompt will change to indicate that you're working within the virtual environment.​

3. Install Dependencies
With the virtual environment activated, install the required packages:​

```
pip install -r requirements.txt
```

This command installs all the dependencies specified in the requirements.txt file.​

4. Run the Flask Application
Start the Flask development server:

```
python app.py
```

By default, the application will be accessible at http://127.0.0.1:5000/ in your web browser.
```
GenomeLREP1/
├── app.py
├── requirements.txt
├── static/
├── templates/
├── README.md
└── .idea/
```

<h1><b>📌 Notes</b></h1>
<li>Ensure that you have Python installed on your system before proceeding.</li>

<li>Using a virtual environment is a best practice to manage project-specific dependencies and avoid conflicts.</li>
