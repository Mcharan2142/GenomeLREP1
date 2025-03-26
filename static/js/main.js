    document.addEventListener('DOMContentLoaded', function() {
        // Setup toggle for method parameters
        const methodCheckboxes = document.querySelectorAll('.method-checkbox');
        methodCheckboxes.forEach(function(checkbox) {
            checkbox.addEventListener('change', function() {
                const methodId = this.id;
                const paramsDiv = document.getElementById(methodId + '-params');
                if (paramsDiv) {
                    paramsDiv.style.display = this.checked ? 'block' : 'none';
                }
            });
        });

        // Handle the 'All' checkbox
        const allCheckbox = document.getElementById('all');
        if (allCheckbox) {
            allCheckbox.addEventListener('change', function() {
                methodCheckboxes.forEach(function(checkbox) {
                    checkbox.checked = allCheckbox.checked;
                    const methodId = checkbox.id;
                    const paramsDiv = document.getElementById(methodId + '-params');
                    if (paramsDiv) {
                        paramsDiv.style.display = allCheckbox.checked ? 'block' : 'none';
                    }
                });
            });
        }

        // Handle number of PCA components change
        const pcaComponentsInput = document.getElementById('pca_n_components');
        if (pcaComponentsInput) {
            pcaComponentsInput.addEventListener('change', function() {
                updatePCOptions(parseInt(this.value));
            });

            // Initialize with default value
            updatePCOptions(parseInt(pcaComponentsInput.value));
        }

        // Function to update PC dropdown options
        function updatePCOptions(numComponents) {
            const pcXSelect = document.getElementById('pc_x');
            const pcYSelect = document.getElementById('pc_y');

            if (!pcXSelect || !pcYSelect) return;

            // Clear existing options
            pcXSelect.innerHTML = '';
            pcYSelect.innerHTML = '';

            // Add new options based on number of components
            for (let i = 1; i <= numComponents; i++) {
                const option = document.createElement('option');
                option.value = `PC${i}`;
                option.textContent = `PC${i}`;

                pcXSelect.appendChild(option.cloneNode(true));
                pcYSelect.appendChild(option.cloneNode(true));
            }

            // Set default selections
            pcXSelect.value = 'PC1';
            pcYSelect.value = numComponents >= 2 ? 'PC2' : 'PC1';
        }

        // Hide file error when a new file is selected
        document.getElementById('file').addEventListener('change', function() {
            var errorDiv = document.querySelector('.alert-danger');
            if (errorDiv) {
                errorDiv.style.display = 'none';
            }
        });
    });