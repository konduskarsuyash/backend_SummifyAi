from django.db import models
from django.contrib.auth.models import User

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    username = models.CharField(max_length=150)  # Store the username
    location = models.CharField(max_length=255, blank=True, null=True)  # Keep it empty initially
    bio = models.TextField(blank=True, null=True)  # Keep it empty initially
    def __str__(self):
        return f"{self.user.username}'s Profile"



class UserStatistics(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='statistics')
    pdfs_summarized = models.IntegerField(default=0)
    quizzes_taken = models.IntegerField(default=0)
    yt_summaries_generated = models.IntegerField(default=0)

    def __str__(self):
        return f"{self.user.username}'s Statistics"


class UserContribution(models.Model):
    # Choices representing types of contributions
    CONTRIBUTION_CHOICES = (
        ('test', 'Test'),  # For tests like the ones in the GeneratedQuiz model
        ('pdf_summary', 'PDF Summary'),  # For PDF summaries like in the PDFSummary model
    )
    
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    contribution_type = models.CharField(max_length=50, choices=CONTRIBUTION_CHOICES)
    date = models.DateField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.contribution_type} on {self.date}"
