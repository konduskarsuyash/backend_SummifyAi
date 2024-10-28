from django.urls import path
from .views import UserProfileDetailView,UserContributionsView,UserStatisticsView
from .views import get_user_id

urlpatterns = [
    path('get-user-id/', get_user_id, name='get_user_id'),

    path('profile/<int:user_id>/', UserProfileDetailView.as_view(), name='user-profile'),  
    path('statistics/<int:user_id>/', UserStatisticsView.as_view(), name='user-statistics'),
    path('contributions/<int:user_id>/', UserContributionsView.as_view(), name='user-contributions'),
]