from django.urls import path
from .views import YouTubeSummaryCreateView, PDFSummarizationView, PDFMindmapView, QuizGeneratorView,VideoUploadView

urlpatterns = [
    path('yt_summarize/', YouTubeSummaryCreateView.as_view(), name='summarize_youtube_video'),
    path('pdf_summarize/', PDFSummarizationView.as_view(), name='pdf_summarize'),
    path('pdf_mindmap/', PDFMindmapView.as_view(), name='pdf_mindmap'),
    path('generate-quiz/', QuizGeneratorView.as_view(), name='generate_quiz'), 
    path("video_summaeizer/",VideoUploadView.as_view(), name='video_summaeizer'),
]
