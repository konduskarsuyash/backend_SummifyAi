from rest_framework import serializers
from .models import UserProfile, UserStatistics, UserContribution

class UserProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserProfile
        fields = ['user', 'username', 'location', 'bio']


class UserStatisticSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserStatistics
        fields = ['pdfs_summarized', 'quizzes_taken', 'yt_summaries_generated']



class UserContributionSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserContribution
        fields = ['user', 'date', 'is_active']
