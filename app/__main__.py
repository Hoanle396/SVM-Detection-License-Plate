from app import app
import app.settings as settings
from routes.route import extractRoute

if __name__ == "__main__":
    extractRoute(app)
    app.run(host=settings.BE_HOST, port=settings.BE_PORT, debug=True)
