from django.apps import AppConfig

class UserProfileConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "user_profile"

    def ready(self):
        # Import the signal handlers when the app is ready
        import user_profile.signals
