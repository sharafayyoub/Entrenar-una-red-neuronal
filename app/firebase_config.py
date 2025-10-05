import firebase_admin
from firebase_admin import credentials, firestore

# Ruta al archivo de credenciales de Firebase
cred = credentials.Certificate("firebase_key.json")


# Inicializar Firebase solo una vez
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

# Crear cliente de Firestore
db = firestore.client()
