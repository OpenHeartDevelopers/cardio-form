# helper_scripts/verify_models.py
# Run after `pip install -e .` (cardio_form is importable from the installed package).

import yaml

from cardio_form.models import ModelManager

def main():
    """
    Verifies that the ModelManager can find all models defined in the manifest.
    """
    print("--- Verifying CardioForm Model Setup ---")
    
    try:
        # We can point to the manifest explicitly if needed, but the default is fine.
        manager = ModelManager()
        print(f"Successfully loaded manifest from: {manager.manifest_path}")
    except Exception as e:
        print(f"FATAL: Could not initialize ModelManager. Error: {e}")
        return

    # Check every version of every model listed in the manifest, so a newly
    # registered version is covered without editing this script.
    with open(manager.manifest_path, 'r') as f:
        manifest = yaml.safe_load(f)
    models_to_check = {key: list(info['versions']) for key, info in manifest.items()}

    all_ok = True
    for model_name, versions in models_to_check.items():
        print(f"\n--- Checking Model: '{model_name}' ---")
        for version in versions:
            try:
                path = manager.get_model_path(model_name, version=version)
                # The get_model_path function already checks if the file exists
                print(f"  ✅ Version '{version}' OK. Found at: {path}")
            except (KeyError, FileNotFoundError) as e:
                print(f"  ❌ Version '{version}' FAILED. Error: {e}")
                all_ok = False
    
    print("\n--- Verification Summary ---")
    if all_ok:
        print("All configured models were found successfully!")
    else:
        print("Some models could not be found. Please check your `models.yaml` and `weights/` directory.")

if __name__ == "__main__":
    main()