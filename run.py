from app import app
from server_config import HOST, PORT, DEBUG

if __name__ == '__main__':
    print(f"Translator App running at http://127.0.0.1:{PORT}")
    # Note: Running on port 80 requires sudo/root privileges on most systems
    app.run(debug=DEBUG, host=HOST, port=PORT)
