from rest_framework import serializers
from django.contrib.auth.models import User
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework import serializers



class RegisterSerializer(serializers.Serializer):
    email = serializers.EmailField()
    username = serializers.CharField()
    password = serializers.CharField()
    
    def validate(self, data):
        if User.objects.filter(username=data['username']).exists():
            raise serializers.ValidationError("Username already exists")
        return data
    
    def create(self, validated_data):
        user = User.objects.create(
            email=validated_data['email'],
            username=validated_data['username']
        )
        
        user.set_password(validated_data['password'])
        user.save()
        return validated_data
    

class LoginSerializer(serializers.Serializer):
    username = serializers.CharField()
    password = serializers.CharField()
    
    def validate(self, data):
        user = User.objects.filter(username=data['username']).first()
        if not user:
            raise serializers.ValidationError("Account not registered")
        if not user.check_password(data['password']):
            raise serializers.ValidationError("Invalid credentials")
        data['user'] = user
        return data
    
    def get_jwt_token(self, data):
        user = data['user']
        
        refresh = RefreshToken.for_user(user)
        
        return {
            "message": "Login success",
            'data': {
                'refresh': str(refresh),
                'access': str(refresh.access_token),
                'username': user.username
            }
        }
        
        
        




        
     