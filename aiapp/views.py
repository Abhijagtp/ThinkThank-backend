from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework.permissions import IsAuthenticated, AllowAny,IsAuthenticatedOrReadOnly
from django.contrib.auth import authenticate
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from .serializers import UserSerializer, LoginSerializer, DocumentSerializer,ChatHistorySerializer,ComparisonHistorySerializer,SavedNoteSerializer,PostSerializer,CommentSerializer
from .models import Document,ComparisonHistory,SavedNote,Post,PostInteraction,Comment
import logging

# Set up logging
logger = logging.getLogger(__name__)

@method_decorator(csrf_exempt, name="dispatch")
class RegisterView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        logger.info(f"Register request data: {request.data}")
        serializer = UserSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            logger.info(f"User created: {user.email}")
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        logger.error(f"Registration errors: {serializer.errors}")
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@method_decorator(csrf_exempt, name="dispatch")
class LoginView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        logger.info(f"Login request data: {request.data}")
        serializer = LoginSerializer(data=request.data)
        if not serializer.is_valid():
            logger.error(f"Login serializer errors: {serializer.errors}")
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        email = serializer.validated_data["email"]
        password = serializer.validated_data["password"]
        user = authenticate(email=email, password=password)
        if user:
            logger.info(f"User authenticated: {user.email}")
            refresh = RefreshToken.for_user(user)
            response = Response({
                "refresh": str(refresh),
                "access": str(refresh.access_token),
                "user": UserSerializer(user).data
            })
            response.set_cookie(
                key="access_token",
                value=str(refresh.access_token),
                httponly=True,
                secure=False,  # False for local development
                samesite="Lax",
                max_age=15 * 60
            )
            return response
        logger.error(f"Authentication failed for email: {email}")
        return Response({"error": "Invalid credentials", "email": email}, status=status.HTTP_401_UNAUTHORIZED)

@method_decorator(csrf_exempt, name="dispatch")
class LogoutView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            refresh_token = request.data["refresh"]
            token = RefreshToken(refresh_token)
            token.blacklist()
            response = Response(status=status.HTTP_205_RESET_CONTENT)
            response.delete_cookie("access_token")
            return response
        except Exception as e:
            logger.error(f"Logout error: {str(e)}")
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

@method_decorator(csrf_exempt, name="dispatch")
class UserView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        serializer = UserSerializer(request.user)
        return Response(serializer.data)
    def put(self, request):
        serializer = UserSerializer(request.user, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class DocumentUploadView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        files = request.FILES.getlist('files')
        if not files:
            print("No files received in request.FILES:", request.FILES)  # Debug
            return Response({'error': 'No files provided'}, status=status.HTTP_400_BAD_REQUEST)

        responses = []
        for file in files:
            allowed_types = ['pdf', 'docx', 'doc', 'csv', 'xlsx']
            file_type = file.name.split('.')[-1].lower()
            if file_type not in allowed_types:
                responses.append({'name': file.name, 'error': 'Unsupported file type'})
                continue
            if file.size > 10 * 1024 * 1024:
                responses.append({'name': file.name, 'error': 'File size exceeds 10MB'})
                continue

            data = {
                'user': request.user,  # Include user in data
                'file': file,
                'name': file.name,
            }
            serializer = DocumentSerializer(data=data, context={'request': request})  # Pass request context
            if serializer.is_valid():
                serializer.save()
                responses.append(serializer.data)
            else:
                responses.append({'name': file.name, 'error': serializer.errors})

        return Response(responses, status=status.HTTP_201_CREATED)

class DocumentListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        documents = Document.objects.filter(user=request.user)
        serializer = DocumentSerializer(documents, many=True, context={'request': request})  # Pass request to context
        return Response(serializer.data)

    def delete(self, request, pk):
        try:
            document = Document.objects.get(pk=pk, user=request.user)
            document.delete()
            return Response(status=status.HTTP_204_NO_CONTENT)
        except Document.DoesNotExist:
            return Response({'error': 'Document not found'}, status=status.HTTP_404_NOT_FOUND)
    


import os
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from .models import Document, CustomUser,ChatHistory
from .serializers import DocumentSerializer
from django.contrib.auth import authenticate
from rest_framework_simplejwt.tokens import RefreshToken
from openai import OpenAI
import PyPDF2
import docx
import pandas as pd
import logging
import json
from django.utils import timezone
from datetime import datetime as Date
import time 
import re

class DocumentAnalysisView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        document_id = request.data.get('document_id')
        user_query = request.data.get('query')
        chat_history = request.data.get('chat_history', [])
        output_format = request.data.get('output_format', 'markdown')
        save_to_notes = request.data.get('save_to_notes', False)
        note_title = request.data.get('note_title', None)
        note_tags = request.data.get('tags', [])
        note_color = request.data.get('color', 'blue')

        if not document_id or not user_query:
            return Response({'error': 'Document ID and query are required'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            document = Document.objects.get(id=document_id, user=request.user)
        except Document.DoesNotExist:
            return Response({'error': 'Document not found'}, status=status.HTTP_404_NOT_FOUND)

        try:
            content = extract_document_content(document)
            logger.info(f"Extracted {len(content)} chars from {document.name}")
        except Exception as e:
            logger.error(f"Error extracting document content: {str(e)}")
            return Response({'error': 'Failed to extract document content'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        estimated_tokens = len(content) // 4 + len(user_query) // 4 + 500
        if estimated_tokens > 1_800_000:
            logger.warning(f"Content too large ({estimated_tokens} tokens). Summarizing...")
            content = content[:50000] + "\n... (truncated for efficiency; full context available if needed)"

        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            logger.error("OPENROUTER_API_KEY is not set in environment variables")
            return Response({'error': 'API configuration error'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

        try:
            system_prompt = (
                "You are a research assistant powered by Grok 4 Fast. Analyze the provided document content "
                "and respond to the user's query with concise, insightful answers. Include key metrics, trends, "
                "or summaries where relevant. Use the full context for accurate analysis. "
                "Format the response as follows:\n"
            )
            if output_format == 'json':
                system_prompt += (
                    "Return a JSON object with three fields: \n"
                    "- 'overview': A brief summary (2-3 sentences).\n"
                    "- 'key_details': An array of key points (each with 'title' and 'description').\n"
                    "- 'insights': An array of insights (each with 'label' and 'value').\n"
                    "Example: ```json\n"
                    "{\"overview\": \"Summary...\", \"key_details\": [{\"title\": \"Point\", \"description\": \"Details...\"}], \"insights\": [{\"label\": \"Trend\", \"value\": \"Data\"}]}"
                    "```"
                )
            else:
                system_prompt += (
                    "Return a markdown-formatted response with:\n"
                    "# Overview\nA brief summary (2-3 sentences).\n"
                    "# Key Details\n- **Title**: Description.\n"
                    "# Insights\n- **Label**: Value.\n"
                    "Use clear headings and concise bullet points."
                )

            messages = [
                {
                    "role": "system",
                    "content": system_prompt + f"\nDocument content: {content}",
                },
                *[
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in chat_history
                ],
                {"role": "user", "content": user_query},
            ]

            # Save user message to ChatHistory
            user_message_id = str(int(time.time() * 1000))
            user_message = {
                'id': user_message_id,
                'type': 'user',
                'content': user_query,
                'timestamp': timezone.now().isoformat(),
                'insights': [],
                'is_json': False,
                'token_usage': None,
            }
            try:
                ChatHistory.objects.create(
                    user=request.user,
                    document=document,
                    message=user_message
                )
                logger.info(f"Saved user message {user_message_id} for user {request.user.email}")
            except Exception as save_error:
                logger.error(f"Failed to save user message: {str(save_error)}")
                # Continue to avoid blocking AI response

            response = client.chat.completions.create(
                model="x-ai/grok-4-fast:free",
                messages=messages,
                max_tokens=1000,
                temperature=0.7,
                stream=False,
                extra_headers={
                    "HTTP-Referer": "https://your-app.com",
                    "X-Title": "AI Research Assistant",
                },
                extra_body={
                    "reasoning": {"enabled": True},
                },
            )

            content = response.choices[0].message.content
            insights = extract_insights(content, output_format)

            # Save AI response to ChatHistory
            ai_message_id = str(int(time.time() * 1000) + 1)
            ai_message = {
                'id': ai_message_id,
                'type': 'ai',
                'content': content,
                'timestamp': timezone.now().isoformat(),
                'insights': insights,
                'is_json': output_format == 'json',
                'token_usage': response.usage.model_dump() if response.usage else None,
            }
            try:
                ChatHistory.objects.create(
                    user=request.user,
                    document=document,
                    message=ai_message
                )
                logger.info(f"Saved AI message {ai_message_id} for user {request.user.email}")
            except Exception as save_error:
                logger.error(f"Failed to save AI message: {str(save_error)}")
                # Continue with response

            # Save to SavedNote if requested
            saved_note = None
            if save_to_notes:
                note_data = {
                    'user': request.user,
                    'title': note_title or f"Analysis of {document.name}",
                    'content': content,
                    'tags': note_tags,
                    'source_document': document,
                    'source_type': 'analysis',
                    'source_id': ai_message_id,
                    'starred': request.data.get('starred', False),
                    'color': note_color,
                }
                try:
                    note = SavedNote.objects.create(**note_data)
                    saved_note = SavedNoteSerializer(note).data
                    logger.info(f"Saved analysis note {note.id} for user {request.user.email}")
                except Exception as e:
                    logger.error(f"Failed to save analysis note: {str(e)}")

            return Response({
                'content': content,
                'insights': insights,
                'token_usage': response.usage.model_dump() if response.usage else None,
                'saved_note': saved_note,
            }, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"OpenRouter API error: {str(e)}")
            return Response({'error': 'Failed to analyze document'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class ChatHistoryView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, document_id):
        try:
            document = Document.objects.get(id=document_id, user=request.user)
            chat_history = ChatHistory.objects.filter(user=request.user, document=document).order_by('created_at')
            serializer = ChatHistorySerializer(chat_history, many=True)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except Document.DoesNotExist:
            return Response({'error': 'Document not found'}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            logger.error(f"Error fetching chat history: {str(e)}")
            return Response({'error': 'Failed to fetch chat history'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)



class DocumentComparisonView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        document1_id = request.data.get('document1_id')
        document2_id = request.data.get('document2_id')
        output_format = request.data.get('output_format', 'markdown')
        save_to_notes = request.data.get('save_to_notes', False)
        note_title = request.data.get('note_title', None)
        note_tags = request.data.get('tags', [])
        note_color = request.data.get('color', 'purple')

        if not document1_id or not document2_id:
            return Response({'error': 'Two document IDs are required'}, status=status.HTTP_400_BAD_REQUEST)
        if document1_id == document2_id:
            return Response({'error': 'Cannot compare the same document'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            document1 = Document.objects.get(id=document1_id, user=request.user)
            document2 = Document.objects.get(id=document2_id, user=request.user)
        except Document.DoesNotExist:
            return Response({'error': 'One or both documents not found'}, status=status.HTTP_404_NOT_FOUND)

        try:
            content1 = extract_document_content(document1)
            content2 = extract_document_content(document2)
            logger.info(f"Extracted {len(content1)} chars from {document1.name}, {len(content2)} chars from {document2.name}")
        except Exception as e:
            logger.error(f"Error extracting document content: {str(e)}")
            return Response({'error': 'Failed to extract document content'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        estimated_tokens = (len(content1) + len(content2)) // 4 + 500
        if estimated_tokens > 1_800_000:
            logger.warning(f"Content too large ({estimated_tokens} tokens). Truncating...")
            content1 = content1[:25000] + "\n... (truncated)"
            content2 = content2[:25000] + "\n... (truncated)"

        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            logger.error("OPENROUTER_API_KEY is not set in environment variables")
            return Response({'error': 'API configuration error'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

        try:
            system_prompt = (
                "You are a research assistant powered by Grok 4 Fast. Compare the two provided documents and provide a concise comparison. "
                "Identify key differences, trends, and insights. Format the response as follows:\n"
            )
            if output_format == 'json':
                system_prompt += (
                    "Return a JSON object with four fields:\n"
                    "- 'summary': A brief summary (2-3 sentences).\n"
                    "- 'keyDifferences': An array of differences (each with 'category', 'doc1Value', 'doc2Value', 'change', 'type' ['positive' or 'negative' or 'neutral'], 'insight').\n"
                    "- 'insights': An array of general insights.\n"
                    "- 'recommendations': An array of actionable recommendations.\n"
                    "Example: ```json\n"
                    "{\"summary\": \"...\", \"keyDifferences\": [{\"category\": \"Metric\", \"doc1Value\": \"Value\", \"doc2Value\": \"Value\", \"change\": \"+/-X%\", \"type\": \"positive\", \"insight\": \"...\"}], \"insights\": [], \"recommendations\": []}"
                    "```"
                )
            else:
                system_prompt += (
                    "Return a markdown-formatted response with:\n"
                    "# Summary\nA brief summary (2-3 sentences).\n"
                    "# Key Differences\n- **Category**: Doc1: Value, Doc2: Value, Change: +/-X% or N/A (Insight).\n"
                    "# Insights\n- Insight text.\n"
                    "# Recommendations\n- Recommendation text.\n"
                    "Use clear headings and concise bullet points. Ensure every key difference includes a Change field, using 'N/A' if no quantitative change applies."
                )

            messages = [
                {
                    "role": "system",
                    "content": (
                        system_prompt + f"\nDocument 1 ({document1.name}): {content1}\n"
                        f"Document 2 ({document2.name}): {content2}"
                    ),
                },
                {"role": "user", "content": "Compare the two documents and highlight key differences, insights, and recommendations."},
            ]

            response = client.chat.completions.create(
                model="x-ai/grok-4-fast:free",
                messages=messages,
                max_tokens=1500,
                temperature=0.7,
                stream=False,
                extra_headers={
                    "HTTP-Referer": "https://your-app.com",
                    "X-Title": "AI Research Assistant",
                },
                extra_body={
                    "reasoning": {"enabled": True},
                },
            )

            content = response.choices[0].message.content
            result = extract_comparison_result(content, output_format)

            # Save comparison result to ComparisonHistory
            comparison_id = str(int(time.time() * 1000))
            comparison_result = {
                'id': comparison_id,
                'summary': result['summary'],
                'keyDifferences': result['keyDifferences'],
                'insights': result['insights'],
                'recommendations': result['recommendations'],
                'timestamp': timezone.now().isoformat(),
                'is_json': output_format == 'json',
                'token_usage': response.usage.model_dump() if response.usage else None,
            }
            try:
                ComparisonHistory.objects.create(
                    user=request.user,
                    document1=document1,
                    document2=document2,
                    result=comparison_result
                )
                logger.info(f"Saved comparison {comparison_id} for user {request.user.email}")
            except Exception as save_error:
                logger.error(f"Failed to save comparison: {str(save_error)}")
                # Continue with response

            # Save to SavedNote if requested
            saved_note = None
            if save_to_notes:
                note_data = {
                    'user': request.user,
                    'title': note_title or f"Comparison of {document1.name} vs {document2.name}",
                    'content': content,
                    'tags': note_tags,
                    'source_document': document1,  # Use first document as source
                    'source_type': 'comparison',
                    'source_id': comparison_id,
                    'starred': request.data.get('starred', False),
                    'color': note_color,
                }
                try:
                    note = SavedNote.objects.create(**note_data)
                    saved_note = SavedNoteSerializer(note).data
                    logger.info(f"Saved comparison note {note.id} for user {request.user.email}")
                except Exception as e:
                    logger.error(f"Failed to save comparison note: {str(e)}")

            return Response({
                'content': content,
                'result': result,
                'token_usage': response.usage.model_dump() if response.usage else None,
                'saved_note': saved_note,
            }, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"OpenRouter API error: {str(e)}")

class ComparisonHistoryView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        try:
            comparisons = ComparisonHistory.objects.filter(user=request.user).order_by('-created_at')
            serializer = ComparisonHistorySerializer(comparisons, many=True)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error fetching comparison history: {str(e)}")
            return Response({'error': 'Failed to fetch comparison history'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)



def extract_comparison_result(content, output_format='markdown'):
    """Extract structured comparison result from AI response."""
    result = {
        'summary': '',
        'keyDifferences': [],
        'insights': [],
        'recommendations': [],
    }
    if output_format == 'json':
        try:
            data = json.loads(content)
            result['summary'] = data.get('summary', '')
            result['keyDifferences'] = data.get('keyDifferences', [])
            result['insights'] = data.get('insights', [])
            result['recommendations'] = data.get('recommendations', [])
            return result
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON response; falling back to markdown parsing")

    # Markdown parsing
    content_lines = content.split('\n')
    current_section = None
    for line in content_lines:
        line = line.strip()
        if line.startswith('# Summary'):
            current_section = 'summary'
            result['summary'] = ''
        elif line.startswith('# Key Differences'):
            current_section = 'keyDifferences'
        elif line.startswith('# Insights'):
            current_section = 'insights'
        elif line.startswith('# Recommendations'):
            current_section = 'recommendations'
        elif line and current_section:
            if current_section == 'summary':
                result['summary'] += line + ' '
            elif current_section == 'keyDifferences' and line.startswith('- **'):
                try:
                    # Match: - **Category**: Doc1: Value, Doc2: Value, [Change: +/-X%] (Insight)
                    match = re.match(r'- \*\*(.*?)\*\*: Doc1: (.*?)(\s*,\s*Doc2: (.*?))?(\s*,?\s*Change: ([+-]?\d+\.?\d*%))?(\s*\((.*?)\))?', line)
                    if match:
                        category = match.group(1).strip()
                        doc1_value = match.group(2).strip()
                        doc2_value = match.group(4).strip() if match.group(4) else ''
                        change = match.group(6).strip() if match.group(6) else 'N/A'
                        insight = match.group(8).strip() if match.group(8) else ''
                        result['keyDifferences'].append({
                            'category': category,
                            'doc1Value': doc1_value,
                            'doc2Value': doc2_value,
                            'change': change,
                            'type': 'positive' if change.startswith('+') else 'negative' if change.startswith('-') else 'neutral',
                            'insight': insight,
                        })
                    else:
                        logger.warning(f"Failed to parse key difference line: {line}")
                except Exception as e:
                    logger.warning(f"Error parsing key difference: {line}, Error: {str(e)}")
            elif current_section in ['insights', 'recommendations'] and line.startswith('- '):
                result[current_section].append(line.strip('- ').strip())
    result['summary'] = result['summary'].strip()
    return result


def extract_document_content(document):
    """Extract text content from a document based on its file type."""
    file_path = document.file.path
    file_type = document.file_type.lower()

    if file_type == 'pdf':
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text() or ''
            return text.strip() or 'No text content extracted'

    elif file_type in ['docx', 'doc']:
        doc = docx.Document(file_path)
        text = '\n'.join([para.text for para in doc.paragraphs if para.text.strip()])
        return text.strip() or 'No text content extracted'

    elif file_type == 'csv':
        df = pd.read_csv(file_path)
        return df.to_string() or 'No text content extracted'

    elif file_type == 'xlsx':
        df = pd.read_excel(file_path)
        return df.to_string() or 'No text content extracted'

    else:
        return 'Unsupported file type'

def extract_insights(response_content, output_format='markdown'):
    """Extract structured insights from AI response."""
    insights = []
    if output_format == 'json':
        try:
            data = json.loads(response_content)
            insights = data.get('insights', [])
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON response; falling back to markdown parsing")
            matches = response_content.split('\n')
            for line in matches:
                if ':' in line and line.strip().startswith('-'):
                    label, value = line.strip('- ').split(':', 1)
                    insights.append({'type': 'trend', 'label': label.strip(), 'value': value.strip()})
    else:
        matches = response_content.split('\n')
        for line in matches:
            if ':' in line and line.strip().startswith('- **'):
                label, value = line.strip('- *').split(':', 1)
                insights.append({'type': 'trend', 'label': label.strip('* '), 'value': value.strip()})
    return insights[:3]


class SavedNoteListCreateView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        try:
            notes = SavedNote.objects.filter(user=request.user).order_by('-created_at')
            serializer = SavedNoteSerializer(notes, many=True)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error fetching saved notes: {str(e)}")
            return Response({'error': 'Failed to fetch saved notes'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def post(self, request):
        try:
            data = request.data.copy()
            data['user'] = request.user.id
            serializer = SavedNoteSerializer(data=data)
            if serializer.is_valid():
                serializer.save()
                logger.info(f"Saved note {serializer.data['id']} for user {request.user.email}")
                return Response(serializer.data, status=status.HTTP_201_CREATED)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            logger.error(f"Error saving note: {str(e)}")
            return Response({'error': 'Failed to save note'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class SavedNoteDetailView(APIView):
    permission_classes = [IsAuthenticated]

    def get_object(self, note_id, user):
        try:
            return SavedNote.objects.get(id=note_id, user=user)
        except SavedNote.DoesNotExist:
            return None

    def get(self, request, note_id):
        note = self.get_object(note_id, request.user)
        if not note:
            return Response({'error': 'Note not found'}, status=status.HTTP_404_NOT_FOUND)
        serializer = SavedNoteSerializer(note)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def put(self, request, note_id):
        note = self.get_object(note_id, request.user)
        if not note:
            return Response({'error': 'Note not found'}, status=status.HTTP_404_NOT_FOUND)
        serializer = SavedNoteSerializer(note, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            logger.info(f"Updated note {note_id} for user {request.user.email}")
            return Response(serializer.data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, note_id):
        note = self.get_object(note_id, request.user)
        if not note:
            return Response({'error': 'Note not found'}, status=status.HTTP_404_NOT_FOUND)
        note.delete()
        logger.info(f"Deleted note {note_id} for user {request.user.email}")
        return Response(status=status.HTTP_204_NO_CONTENT)
    


# posts and interactions

class PostListCreateView(APIView):
    permission_classes = [IsAuthenticatedOrReadOnly]

    def get(self, request):
        posts = Post.objects.all()
        serializer = PostSerializer(posts, many=True, context={'request': request})
        return Response(serializer.data)

    def post(self, request):
        serializer = PostSerializer(data=request.data, context={'request': request})
        if serializer.is_valid():
            serializer.save(user=request.user)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class PostInteractionView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, post_id):
        post = Post.objects.get(id=post_id)
        interaction, created = PostInteraction.objects.get_or_create(user=request.user, post=post)
        
        action = request.data.get('action')  # 'like', 'unlike', 'save', 'unsave'
        if action == 'like':
            if not interaction.liked:
                interaction.liked = True
                post.likes += 1
            interaction.save()
            post.save()
        elif action == 'unlike':
            if interaction.liked:
                interaction.liked = False
                post.likes -= 1
            interaction.save()
            post.save()
        elif action == 'save':
            interaction.saved = True
            interaction.save()
        elif action == 'unsave':
            interaction.saved = False
            interaction.save()
        
        return Response(PostSerializer(post, context={'request': request}).data)

class CommentListCreateView(APIView):
    permission_classes = [IsAuthenticatedOrReadOnly]

    def get(self, request, post_id):
        try:
            comments = Comment.objects.filter(post_id=post_id)
            serializer = CommentSerializer(comments, many=True)
            logger.info(f"Fetched {len(comments)} comments for post_id={post_id}")
            return Response(serializer.data)
        except Exception as e:
            logger.error(f"Error fetching comments for post_id={post_id}: {str(e)}")
            return Response({"error": "Failed to fetch comments"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def post(self, request, post_id):
        try:
            post = Post.objects.get(id=post_id)
            serializer = CommentSerializer(data=request.data)
            if serializer.is_valid():
                serializer.save(user=request.user, post=post)
                logger.info(f"Comment created for post_id={post_id} by user={request.user.email}")
                return Response(serializer.data, status=status.HTTP_201_CREATED)
            logger.error(f"Comment creation failed for post_id={post_id}: {serializer.errors}")
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        except Post.DoesNotExist:
            logger.error(f"Post not found: post_id={post_id}")
            return Response({"error": "Post not found"}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            logger.error(f"Error creating comment for post_id={post_id}: {str(e)}")
            return Response({"error": "Failed to create comment"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
class UserCommentsView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        comments = Comment.objects.filter(user=request.user)
        serializer = CommentSerializer(comments, many=True)
        return Response(serializer.data)
    



# health check -
from django.http import JsonResponse

def health_check(request):
    return JsonResponse({"status": "ok"}, status=200)


