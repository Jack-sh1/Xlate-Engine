import os
import sass

def compile_scss():
    """
    Compiles SCSS files in the static directory to CSS.
    """
    try:
        # Get the absolute path of the current file's directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        scss_file = os.path.join(base_dir, 'static', 'style.scss')
        css_file = os.path.join(base_dir, 'static', 'style.css')
        
        if os.path.exists(scss_file):
            print(f"Compiling {scss_file} to {css_file}...")
            # Ensure the directory exists (though it should)
            os.makedirs(os.path.dirname(css_file), exist_ok=True)
            
            with open(css_file, 'w') as f:
                f.write(sass.compile(filename=scss_file))
            print("SCSS compilation successful.")
        else:
            print(f"Warning: SCSS file not found at {scss_file}")
    except Exception as e:
        print(f"SCSS compilation failed: {e}")

if __name__ == "__main__":
    compile_scss()
