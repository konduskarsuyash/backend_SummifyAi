from django.contrib import admin

# Register your models here.
from .models import YouTubeSummary,PDFSummary,GeneratedQuiz,PDFMindMap

admin.site.register(YouTubeSummary)
admin.site.register(PDFSummary)
admin.site.register(GeneratedQuiz)
admin.site.register(PDFMindMap)