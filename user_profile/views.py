from rest_framework import viewsets
from .models import UserProfile, UserStatistics, UserContribution
from .serializers import UserProfileSerializer, UserStatisticSerializer, UserContributionSerializer
from rest_framework import generics
from rest_framework.permissions import IsAuthenticated
from .models import UserProfile  # Make sure to import 
from rest_framework.exceptions import ValidationError
from rest_framework.response import Response
from rest_framework import status
from datetime import date, timedelta
from rest_framework.views import APIView
from django.contrib.auth.models import User
from rest_framework.exceptions import NotFound
from rest_framework.decorators import api_view, permission_classes


class UserProfileDetailView(generics.RetrieveUpdateAPIView):
    queryset = UserProfile.objects.all()
    serializer_class = UserProfileSerializer
    permission_classes = [IsAuthenticated]
    lookup_field = 'user_id'  # Use the user_id for lookups

    def get_queryset(self):
        # Only allow the logged-in user to retrieve their profile
        return UserProfile.objects.filter(user=self.request.user)

    def get(self, request, user_id):
        # Ensure the user_id corresponds to the logged-in user's ID
        if request.user.id != user_id:
            return Response({"error": "Unauthorized access."}, status=status.HTTP_403_FORBIDDEN)

        user_profile = self.get_queryset().filter(user_id=user_id).first()
        if user_profile:
            serializer = UserProfileSerializer(user_profile)
            return Response(serializer.data, status=status.HTTP_200_OK)
        return Response({"error": "User profile not found."}, status=status.HTTP_404_NOT_FOUND)

    def post(self, request, *args, **kwargs):
        user_id = self.kwargs['user_id']

        # Check if the user already has a profile
        if UserProfile.objects.filter(user_id=user_id).exists():
            raise ValidationError({"detail": "Profile for this user already exists."})

        # Create the profile for the user
        serializer = self.get_serializer(data=request.data)
        if serializer.is_valid():
            serializer.save(user=request.user)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def update(self, request, *args, **kwargs):
        # Get the user profile to update
        user_profile = self.get_object()
        user = request.user  # The associated User object

        # Handle username updates if present in the request data
        username = request.data.get('username')
        if username:
            # Check if the username is different from the current one
            if user.username != username:
                # Ensure the username is not already taken
                if User.objects.filter(username=username).exists():
                    return Response({"error": "This username is already taken."}, status=status.HTTP_400_BAD_REQUEST)
                user.username = username
                user.save()

        # Update the UserProfile model fields
        serializer = self.get_serializer(user_profile, data=request.data, partial=True)  # Allow partial updates

        # Validate and save the updated profile
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class UserStatisticsView(generics.RetrieveAPIView):
    permission_classes = [IsAuthenticated]
    serializer_class = UserStatisticSerializer

    def get_object(self):
        user_id = self.kwargs.get('user_id')
        try:
            user = User.objects.get(id=user_id)
            user_statistics, created = UserStatistics.objects.get_or_create(user=user)
            return user_statistics
        except User.DoesNotExist:
            raise NotFound("User not found.")

# Adjust this view to include user data as well
class UserContributionsView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, user_id):
        user = User.objects.get(id=user_id)
        contributions = get_contributions_for_last_n_days(user, days=30)
        
        response_data = {
            "username": user.username,  # Include the username
            "contributions": {
                str(day): {
                    'highlight': bool(contributions[day]),
                    'contributions': contributions[day],
                }
                for day in contributions
            }
        }
        return Response(response_data)



@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_user_id(request):
    user_id = request.user.id
    return Response({'user_id': user_id})
    
def get_contributions_for_last_n_days(user, days=30):
    today = date.today()
    start_date = today - timedelta(days=days)
    contributions = UserContribution.objects.filter(user=user, date__range=[start_date, today])
    
    # Create a dictionary to store contributions by date
    contributions_by_day = {today - timedelta(days=i): [] for i in range(days)}
    for contribution in contributions:
        contributions_by_day[contribution.date].append(contribution.contribution_type)

    return contributions_by_day