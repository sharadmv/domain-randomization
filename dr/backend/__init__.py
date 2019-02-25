from dr.backend.dart import DartBackend

BACKENDS = {
    'dart': DartBackend(),
    'mujoco': None,
}
def get_backend(backend_name):
    if backend_name not in BACKENDS:
        raise Exception(f"Backend {backend_name} not found")
    return BACKENDS[backend_name]
