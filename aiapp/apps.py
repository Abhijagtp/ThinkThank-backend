from django.apps import AppConfig


class AiappConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "aiapp"

    def ready(self):
        import aiapp.signals
