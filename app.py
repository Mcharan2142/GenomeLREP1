from flask import Flask, render_template, request, send_file, jsonify
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import io
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.decomposition import PCA

app = Flask(__name__)

if not os.path.exists('static'):
    os.makedirs('static')

if not os.path.exists('static/css'):
    os.makedirs('static/css')


@app.route('/')
def index():
    return render_template('index.html', active_page='home')

# Services/Analysis page route
@app.route('/services')
def services():
    return render_template('services.html', active_page='services')

# Login page route
@app.route('/login')
def login():
    return render_template('login.html', active_page='login')

# Signup page route
@app.route('/signup')
def signup():
    return render_template('signup.html', active_page='signup')


@app.route('/about')
def about():
    return render_template('about.html', active_page='about')

# Analysis form processing
@app.route('/analyze', methods=['POST'])
def analyze():

    lasso_html, ridge_html, elasticnet_html, pca_html, pca_plot_path, error, pca = [None]*7

    form_submitted = False
    pca_components = []

    if request.method == 'POST':
        form_submitted = True

        # Check if a file was uploaded
        if 'file' not in request.files:
            return render_template('services.html',
                                   error='No file uploaded',
                                   form_submitted=False,
                                   active_page='services')

        file = request.files['file']

        if file.filename == '':
            return render_template('services.html',
                                   error='No file selected',
                                   form_submitted=False,
                                   active_page='services')

        # Read the CSV file into a pandas DataFrame
        try:
            df = pd.read_csv(file)
        except Exception as e:
            return render_template('services.html',
                                   error=f'Error reading file: {str(e)}',
                                   form_submitted=False,
                                   active_page='services')


        gene_names = df.iloc[:, 0]  # Store gene names
        X = df.iloc[:, 1:-1]  # Features: all columns except the first and last
        y = df.iloc[:, -1]  # Target: last column

        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

        # Determine which methods to use
        use_all = 'all' in request.form
        use_lasso = 'lasso' in request.form or use_all
        use_ridge = 'ridge' in request.form or use_all
        use_elasticnet = 'elasticnet' in request.form or use_all
        use_pca = 'pca' in request.form or use_all

        # Get hyperparameters from form
        lasso_alpha = float(request.form.get('lasso_alpha', 0.1))
        ridge_alpha = float(request.form.get('ridge_alpha', 0.1))
        elasticnet_alpha = float(request.form.get('elasticnet_alpha', 0.1))
        elasticnet_l1_ratio = float(request.form.get('elasticnet_l1_ratio', 0.5))

        # Get PCA parameters
        pca_n_components = int(request.form.get('pca_n_components', 2))
        pc_x = request.form.get('pc_x', 'PC1')
        pc_y = request.form.get('pc_y', 'PC2')

        # Perform regression based on selected methods
        results = {}
        pca_results = None

        if use_lasso:
            lasso = Lasso(alpha=lasso_alpha)
            lasso.fit(X_scaled_df, y)
            s_no = range(1, len(X.columns) + 1)
            lasso_df = pd.DataFrame({'Gene Name': X.columns, 'Lasso': lasso.coef_})
            lasso_html = lasso_df.to_html(classes='table table-striped table-bordered', index=False)
            results['Lasso'] = lasso.coef_

        if use_ridge:
            ridge = Ridge(alpha=ridge_alpha)
            ridge.fit(X_scaled_df, y)
            s_no = range(1, len(X.columns) + 1)
            ridge_df = pd.DataFrame({'Gene Name': X.columns, 'Ridge': ridge.coef_})
            ridge_html = ridge_df.to_html(classes='table table-striped table-bordered', index=False)
            results['Ridge'] = ridge.coef_

        if use_elasticnet:
            elasticnet = ElasticNet(alpha=elasticnet_alpha, l1_ratio=elasticnet_l1_ratio)
            elasticnet.fit(X_scaled_df, y)
            s_no = range(1, len(X.columns) + 1)
            elasticnet_df = pd.DataFrame({'Gene Name': X.columns, 'ElasticNet': elasticnet.coef_})
            elasticnet_html = elasticnet_df.to_html(classes='table table-striped table-bordered', index=False)
            results['ElasticNet'] = elasticnet.coef_

        if use_pca:
            # Apply PCA with specified number of components
            n_components = min(pca_n_components, X_scaled.shape[1])  # Ensure n_components doesn't exceed # of features
            pca = PCA(n_components=n_components)
            pca_result = pca.fit_transform(X_scaled)

            # Create column names for all components
            pc_columns = [f'PC{i + 1}' for i in range(n_components)]
            pca_components = pc_columns  # Store for dropdown options

            # Create DataFrame for PCA results
            pca_df = pd.DataFrame(data=pca_result, columns=pc_columns)

            # Add sample names if available (using row indices if not)
            if 'Sample' in df.columns:
                pca_df['Sample'] = df['Sample']
            else:
                pca_df['Sample'] = [f'Sample {i + 1}' for i in range(len(pca_df))]

            # Prepare the table HTML - include all components in the table
            pca_table_df = pd.DataFrame({
                'Sample': pca_df['Sample'],
                **{col: pca_df[col] for col in pc_columns}
            })
            pca_html = pca_table_df.to_html(classes='table table-striped table-bordered', index=False)

            # Store the PCA results for download
            pca_results = pca_table_df

            # Determine which PCs to plot
            if pc_x not in pc_columns: pc_x = pc_columns[0]  # Default to PC1 if invalid
            if pc_y not in pc_columns: pc_y = pc_columns[1] if len(pc_columns) > 1 else pc_columns[
                0]  # Default to PC2 if possible

            # Create PCA plot with selected components
            plt.figure(figsize=(10, 8))
            plt.scatter(pca_df[pc_x], pca_df[pc_y], alpha=0.8, edgecolors='w')

            # Add labels with explained variance
            x_index = int(pc_x.replace('PC', '')) - 1  # Get index from PC label
            y_index = int(pc_y.replace('PC', '')) - 1
            plt.xlabel(f'{pc_x} ({pca.explained_variance_ratio_[x_index] * 100:.2f}%)')
            plt.ylabel(f'{pc_y} ({pca.explained_variance_ratio_[y_index] * 100:.2f}%)')
            plt.title(f'PCA: {pc_x} vs {pc_y}')

            # Add grid
            plt.grid(True, linestyle='--', alpha=0.7)

            # Save plot as a static file
            plot_filename = 'pca_plot.png'
            plot_path = os.path.join('static', plot_filename)
            plt.savefig(plot_path)
            plt.close()

            # Pass the plot path to the template
            pca_plot_path = '/' + plot_path

        # Store the results and corresponding DataFrames in the app object for download
        app.results = results
        app.X_columns = X.columns
        app.pca_results = pca_results
        # Store the explained variance for later use in plot updates
        app.pca_explained_variance = pca.explained_variance_ratio_

    return render_template('services.html',
                           active_page='services',
                           lasso_html=lasso_html,
                           ridge_html=ridge_html,
                           elasticnet_html=elasticnet_html,
                           pca_html=pca_html,
                           pca_plot_path=pca_plot_path,
                           pca_components=pca_components,
                           form_submitted=form_submitted,
                           error=error)


@app.route('/download/<filetype>')
def download_results(filetype):
    # Retrieve data from the app object
    results = getattr(app, 'results', None)
    X_columns = getattr(app, 'X_columns', None)
    pca_results = getattr(app, 'pca_results', None)

    if results is None or X_columns is None:
        return "No results to download.", 400

    # Build combined coefficients DataFrame
    coefficients_df = pd.DataFrame(results, index=X_columns)

    try:
        if filetype == 'xlsx':
            # Create an Excel file in memory
            excel_file = io.BytesIO()
            with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
                # Write regression results to Excel
                coefficients_df.to_excel(writer, sheet_name='Regression Results', index=True)

                # Write PCA results to Excel if available
                if pca_results is not None:
                    pca_results.to_excel(writer, sheet_name='PCA Results', index=False)

            excel_file.seek(0)

            return send_file(excel_file, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                             download_name='regression_results.xlsx', as_attachment=True)

        else:
            return "Invalid file type specified.", 400

    except Exception as e:
        return f"Error creating or sending file: {str(e)}", 500


@app.route('/download_pca_table')
def download_pca_table():
    """Download the PCA table as an Excel file."""
    # Retrieve PCA data from the app object
    pca_results = getattr(app, 'pca_results', None)

    if pca_results is None:
        return "No PCA results to download.", 400

    try:
        # Create an Excel file in memory
        excel_file = io.BytesIO()
        with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
            pca_results.to_excel(writer, sheet_name='PCA Results', index=False)
        excel_file.seek(0)

        return send_file(excel_file,
                         mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                         download_name='pca_results.xlsx',
                         as_attachment=True)
    except Exception as e:
        return f"Error creating or sending file: {str(e)}", 500


@app.route('/download_pca_plot')
def download_pca_plot():
    """Download the PCA plot as a PNG file."""
    plot_path = os.path.join('static', 'pca_plot.png')
    try:
        return send_file(plot_path, mimetype='image/png', as_attachment=True, download_name="pca_plot.png")
    except Exception as e:
        return f"Error: {str(e)}", 500


@app.route('/update_pca_plot', methods=['GET'])
def update_pca_plot():
    """Generate an updated PCA plot based on selected components."""
    try:
        # Get selected components
        pc_x = request.args.get('pc_x', 'PC1')
        pc_y = request.args.get('pc_y', 'PC2')

        # Retrieve stored PCA results
        pca_results = getattr(app, 'pca_results', None)

        if pca_results is None:
            return jsonify({'success': False, 'error': 'No PCA results available'})

        # Create a new plot using the selected components
        plt.figure(figsize=(10, 8))

        # Extract the numeric indices from the PC names
        x_index = int(pc_x.replace('PC', '')) - 1
        y_index = int(pc_y.replace('PC', '')) - 1

        # Plot the selected components
        plt.scatter(pca_results[pc_x], pca_results[pc_y], alpha=0.8, edgecolors='w')

        # Add labels
        explained_var = getattr(app, 'pca_explained_variance', None)
        if explained_var is not None:
            plt.xlabel(f'{pc_x} ({explained_var[x_index] * 100:.2f}%)')
            plt.ylabel(f'{pc_y} ({explained_var[y_index] * 100:.2f}%)')
        else:
            plt.xlabel(pc_x)
            plt.ylabel(pc_y)

        plt.title(f'PCA: {pc_x} vs {pc_y}')
        plt.grid(True, linestyle='--', alpha=0.7)

        # Save the plot
        plot_filename = 'pca_plot.png'
        plot_path = os.path.join('static', plot_filename)
        plt.savefig(plot_path)
        plt.close()

        return jsonify({'success': True, 'plot_path': '/' + plot_path})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    # Ensure a secret key is set (required for session usage)
    app.secret_key = os.urandom(24)  # Generate a random secret key
    app.run(debug=True)
