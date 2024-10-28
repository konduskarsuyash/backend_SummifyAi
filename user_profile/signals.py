from django.db.models.signals import post_save
from django.contrib.auth.models import User
from .models import UserStatistics
from django.dispatch import receiver
from django.contrib.auth.models import User
from .models import UserProfile
def create_user_statistics(sender, instance, created, **kwargs):
    if created:
        UserStatistics.objects.create(user=instance)

post_save.connect(create_user_statistics, sender=User)


@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.create(user=instance, username=instance.username)