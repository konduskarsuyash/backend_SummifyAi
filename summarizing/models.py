from django.db import models
from django.contrib.auth.models import User

class YouTubeSummary(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="summaries")
    youtube_url = models.URLField(max_length=500)
    transcript = models.TextField()
    summary = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Summary by {self.user.username} for {self.youtube_url}"


class PDFSummary(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="pdf_summaries")
    pdf_file = models.FileField(upload_to='pdfs/')
    start_page_number = models.IntegerField()
    end_page_number = models.IntegerField()
    summary = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Summary by {self.user.username} for Pages {self.start_page_number} to {self.end_page_number} of {self.pdf_file.name}"

    


class GeneratedQuiz(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    pdf_summary = models.ForeignKey(PDFSummary, on_delete=models.CASCADE)
    quiz_data = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Quiz {self.id} for {self.user.username}"
    
    

class PDFMindMap(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="mindmaps")
    pdf_file = models.FileField(upload_to='pdfs/')
    mindmap_json = models.JSONField()  # Store the mind map as JSON
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Mindmap by {self.user.username} for {self.pdf_file.name}"
    

class Video(models.Model):
    video_file = models.FileField(upload_to='uploads/')
    audio_file = models.FileField(upload_to='audio/', null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Video {self.id}: {self.video_file.name}"

