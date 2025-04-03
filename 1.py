import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import os
from datetime import datetime, timedelta
import random
import base64
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter

# Set page config
st.set_page_config(
    page_title="National Poster Presentation Analysis",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
   
    .metric-label {
        font-size: 1.2rem;
        font-weight: bold;
        color: #555;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .highlight {
        background-color: #;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #1E88E5;
    }
    /* Add these styles to your existing CSS */
    .image-container {
        padding: 10px;
        background-color: #f0f0f0;
        border-radius: 10px;
        border: 1px solid #ddd;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .filter-name {
        color: #0D47A1;
        font-weight: bold;
        font-size: 1.1rem;
        background-color: rgba(255,255,255,0.8);
        padding: 3px 8px;
        border-radius: 4px;
        margin-bottom: 5px;
        display: inline-block;
    }
    .download-btn {
        margin-top: 12px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Function to generate dataset with configurable parameters
def generate_dataset(num_participants=400, start_date=datetime(2025, 3, 15), num_days=4):
    # Fixed data
    states = ['Maharashtra', 'Tamil Nadu', 'Karnataka', 'Gujarat', 'Delhi', 
              'Uttar Pradesh', 'West Bengal', 'Telangana', 'Kerala', 'Punjab']
    
    tracks = ['AI & Machine Learning', 'Sustainable Engineering', 
              'Biomedical Innovations', 'Smart Infrastructure']
    
    colleges = [
        'IIT Bombay', 'IIT Delhi', 'NIT Trichy', 'BITS Pilani', 'VIT Vellore',
        'Anna University', 'COEP Pune', 'IIIT Hyderabad', 'Jadavpur University', 'DTU Delhi',
        'Manipal Institute of Technology', 'PSG College of Technology', 'SRM University', 
        'Amrita University', 'PEC Chandigarh', 'NSIT Delhi', 'BMS College of Engineering',
        'College of Engineering Guindy', 'VJTI Mumbai', 'IIIT Bangalore'
    ]
    
    # Feedback templates
    positive_templates = [
        "The {track} session was excellent. {aspect} was particularly impressive.",
        "I really enjoyed the {track} presentations. The {aspect} was outstanding.",
        "Great work on {track}! The {aspect} exceeded my expectations.",
        "The {track} posters were very informative. {aspect} was the highlight.",
        "Excellent organization of the {track} session. {aspect} was well-executed."
    ]
    
    neutral_templates = [
        "The {track} presentations were satisfactory. {aspect} could be improved.",
        "The {track} session was decent. {aspect} was average.",
        "The {track} posters were informative but {aspect} needs more attention.",
        "Adequate presentation in the {track} track. {aspect} was as expected.",
        "The {track} session met basic standards. {aspect} was sufficient."
    ]
    
    negative_templates = [
        "The {track} session needed improvement. {aspect} was lacking.",
        "I was disappointed with the {track} presentations. {aspect} was below expectations.",
        "The {track} posters could be better organized. {aspect} was confusing.",
        "The {track} track had significant issues. {aspect} requires complete rethinking.",
        "Poor execution in the {track} session. {aspect} was inadequate."
    ]
    
    aspects = {
        'AI & Machine Learning': [
            'The algorithm demonstrations', 'The technical explanations', 
            'The practical applications', 'The research methodology', 
            'The data visualization'
        ],
        'Sustainable Engineering': [
            'The eco-friendly designs', 'The energy efficiency models', 
            'The sustainability metrics', 'The green technology innovations', 
            'The environmental impact analysis'
        ],
        'Biomedical Innovations': [
            'The medical device prototypes', 'The healthcare solutions', 
            'The bioethical considerations', 'The clinical trial data', 
            'The patient-centered approach'
        ],
        'Smart Infrastructure': [
            'The urban planning concepts', 'The IoT implementations', 
            'The smart city models', 'The infrastructure resilience', 
            'The connectivity solutions'
        ]
    }
    
    # Generate data
    data = []
    participant_id = 1
    
    for day in range(1, num_days+1):
        current_date = start_date + timedelta(days=day-1)
        date_str = current_date.strftime('%Y-%m-%d')
        
        # Distribute participants unevenly across tracks and days
        track_distribution = [random.randint(20, 30) for _ in range(len(tracks))]
        track_distribution = [round(t/sum(track_distribution) * 100) for t in track_distribution]
        
        for track_idx, track in enumerate(tracks):
            # Number of participants for this track on this day
            num_track_participants = round(track_distribution[track_idx] / 100 * (num_participants / num_days))
            
            for _ in range(num_track_participants):
                if participant_id > num_participants:  # Ensure we don't exceed requested participants
                    break
                    
                state = random.choice(states)
                college = random.choice(colleges)
                
                # Generate scores between 60 and 100
                technical_score = random.randint(60, 100)
                presentation_score = random.randint(60, 100)
                innovation_score = random.randint(60, 100)
                
                # Calculate total score as weighted average
                total_score = (technical_score * 0.4) + (presentation_score * 0.3) + (innovation_score * 0.3)
                
                # Determine sentiment based on total score
                if total_score >= 85:
                    sentiment = "positive"
                    templates = positive_templates
                elif total_score >= 70:
                    sentiment = "neutral"
                    templates = neutral_templates
                else:
                    sentiment = "negative"
                    templates = negative_templates
                
                # Generate feedback
                template = random.choice(templates)
                aspect = random.choice(aspects[track])
                feedback = template.format(track=track, aspect=aspect)
                
                data.append({
                    'Participant_ID': participant_id,
                    'Date': date_str,
                    'Track': track,
                    'State': state,
                    'College': college,
                    'Technical_Score': technical_score,
                    'Presentation_Score': presentation_score,
                    'Innovation_Score': innovation_score,
                    'Total_Score': round(total_score, 2),
                    'Feedback': feedback
                })
                
                participant_id += 1
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    return df

# Function to create sample images for demo with improved visibility
def create_sample_images():
    # Create a directory for images if it doesn't exist
    if not os.path.exists("event_images"):
        os.makedirs("event_images")
    
    tracks = ['AI_ML', 'Sustainable_Eng', 'Biomedical', 'Smart_Infrastructure']
    days = ['Day1', 'Day2', 'Day3', 'Day4']
    
    # Enhanced colors for better visibility
    track_colors = {
        'AI_ML': (200, 50, 50),              # Deep Red
        'Sustainable_Eng': (50, 150, 50),    # Darker Green
        'Biomedical': (50, 50, 200),         # Deeper Blue
        'Smart_Infrastructure': (200, 150, 0) # Darker Yellow/Gold
    }
    
    images = {}
    
    for day in days:
        day_images = {}
        for track in tracks:
            # Create a colored image with dark background for better contrast
            img = np.ones((300, 400, 3), dtype=np.uint8) * 240  # Slightly off-white background
            
            # Add colored rectangle with darker colors for better visibility
            color = track_colors[track]
            cv2.rectangle(img, (50, 50), (350, 250), color, -1)
            
            # Add text with better contrast - black text with white outline
            cv2.putText(img, track, (70, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)  # Thicker black outline
            cv2.putText(img, track, (70, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)  # White text
            
            cv2.putText(img, day, (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)  # Thicker black outline
            cv2.putText(img, day, (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # White text
            
            # Add a border to the image for definition
            cv2.rectangle(img, (20, 20), (380, 280), (100, 100, 100), 2)
            
            # Save image
            filename = f"event_images/{day}_{track}.jpg"
            cv2.imwrite(filename, img)
            
            # Store image path
            day_images[track] = filename
        
        images[day] = day_images
    
    return images

# Function to apply image filter
def apply_filter(image, filter_name='original'):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if filter_name == 'grayscale':
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif filter_name == 'blur':
        return cv2.GaussianBlur(image, (15, 15), 0)
    elif filter_name == 'edges':
        return cv2.Canny(image, 100, 200)
    elif filter_name == 'enhance':
        enhancer = ImageEnhance.Contrast(pil_image)
        enhanced = enhancer.enhance(2.0)
        return cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
    elif filter_name == 'sharpen':
        sharpened = pil_image.filter(ImageFilter.SHARPEN)
        return cv2.cvtColor(np.array(sharpened), cv2.COLOR_RGB2BGR)
    else:
        return image

def get_image_download_link(img, filename, text):
    """Generate a link to download the image"""
    buffered = BytesIO()
    Image.fromarray(img).save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}" download="{filename}">{text}</a>'
    return href

# Main function
def main():
    st.markdown('<h1 class="main-header">🎓 National Poster Presentation Event Analysis</h1>', unsafe_allow_html=True)
    
    # Create sidebar menu with better styling
    st.sidebar.markdown('<div style="text-align: center; padding: 10px; background-color: #e3f2fd; border-radius: 5px; margin-bottom: 20px;"><h2 style="color: #0D47A1;">Navigation</h2></div>', unsafe_allow_html=True)
    
    menu = st.sidebar.radio(
        "",
        ["Home", "Dashboard", "Text Analysis", "Image Gallery"],
        captions=["Generate or upload dataset", "View analytics", "Analyze feedback", "View and edit images"]
    )
    
    # Display information based on menu selection
    if menu == "Home":
        # Only generate data when requested
        if 'data' not in st.session_state:
            st.markdown('<div class="card highlight">', unsafe_allow_html=True)
            st.markdown('<h2 class="sub-header">Generate Dataset</h2>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                num_participants = st.slider("Number of Participants", 100, 1000, 400, 50)
                event_year = st.slider("Event Year", 2023, 2026, 2025)
                
            with col2:
                event_month = st.selectbox("Event Month", list(range(1, 13)), index=2)  # Default to March (index 2)
                event_day = st.selectbox("Event Day", list(range(1, 29)), index=14)  # Default to 15th
                num_days = st.slider("Event Duration (Days)", 1, 7, 4)
            
            start_date = datetime(event_year, event_month, event_day)
            
            if st.button("Generate Dataset", key="generate_dataset"):
                with st.spinner("Generating dataset..."):
                    st.session_state.data = generate_dataset(
                        num_participants=num_participants,
                        start_date=start_date,
                        num_days=num_days
                    )
                    st.session_state.images = create_sample_images()
                    st.success(f"Dataset generated with {num_participants} participants!")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<h2 class="sub-header">Upload Existing Dataset</h2>', unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.data = df
                    st.session_state.images = create_sample_images()
                    st.success("Dataset uploaded successfully!")
                except Exception as e:
                    st.error(f"Error uploading file: {e}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Sample Data Preview
            st.markdown('<h2 class="sub-header">Sample Data Preview</h2>', unsafe_allow_html=True)
            st.markdown(
                """
                <div class="highlight">
                This application analyzes data from a National Poster Presentation Event. 
                Generate a dataset or upload your own data to get started.
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        # If data exists, display overview
        if 'data' in st.session_state:
            df = st.session_state.data
            
            st.markdown('<h2 class="sub-header">📋 Event Overview</h2>', unsafe_allow_html=True)
            
            # Event metrics in a nicer layout
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f'<p class="metric-label">Total Participants</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="metric-value">{len(df)}</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f'<p class="metric-label">Number of Colleges</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="metric-value">{df["College"].nunique()}</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col3:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f'<p class="metric-label">Presentation Tracks</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="metric-value">{df["Track"].nunique()}</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col4:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f'<p class="metric-label">Event Duration</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="metric-value">{df["Date"].nunique()} Days</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Top Performers
            st.markdown('<h2 class="sub-header">🏆 Top Performers</h2>', unsafe_allow_html=True)
            top_performers = df.nlargest(5, 'Total_Score')
            
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.dataframe(top_performers[['Participant_ID', 'College', 'Track', 'Total_Score']], use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Sample Data
            st.markdown('<h2 class="sub-header">📊 Sample Data</h2>', unsafe_allow_html=True)
            
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.dataframe(df.head(10), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Download dataset
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Complete Dataset",
                data=csv,
                file_name="poster_presentation_data.csv",
                mime="text/csv",
            )
            
            # Option to regenerate data
            if st.button("Generate New Dataset", key="regenerate_dataset"):
                st.session_state.pop('data', None)
                st.session_state.pop('images', None)
                st.rerun()
        
    elif menu == "Dashboard":
        if 'data' not in st.session_state:
            st.warning("Please generate or upload a dataset from the Home page first!")
            return
            
        df = st.session_state.data
        
        st.markdown('<h2 class="sub-header">📈 Event Analytics Dashboard</h2>', unsafe_allow_html=True)
        
        # Filters in sidebar
        st.sidebar.markdown('<div style="text-align: center; padding: 10px; background-color: #e3f2fd; border-radius: 5px; margin: 20px 0;"><h3 style="color: #0D47A1;">Dashboard Filters</h3></div>', unsafe_allow_html=True)
        
        # Filter by track
        selected_tracks = st.sidebar.multiselect(
            "Select Tracks",
            options=df['Track'].unique(),
            default=df['Track'].unique()
        )
        
        # Filter by state
        selected_states = st.sidebar.multiselect(
            "Select States",
            options=df['State'].unique(),
            default=df['State'].unique()[:5]  # Default to first 5 states
        )
        
        # Filter by college
        selected_colleges = st.sidebar.multiselect(
            "Select Colleges",
            options=df['College'].unique(),
            default=df['College'].unique()[:5]  # Default to first 5 colleges
        )
        
        # Filter data
        filtered_df = df[
            (df['Track'].isin(selected_tracks)) &
            (df['State'].isin(selected_states)) &
            (df['College'].isin(selected_colleges))
        ]
        
        if filtered_df.empty:
            st.warning("No data available with the selected filters.")
        else:
            # Create tabs for different chart categories
            tab1, tab2, tab3 = st.tabs(["Participation Analysis", "Geographic Distribution", "Performance Metrics"])
            
            with tab1:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Chart 1: Participation by Track
                    st.markdown('<h3 style="color: #0D47A1;">📊 Participation by Track</h3>', unsafe_allow_html=True)
                    track_counts = filtered_df['Track'].value_counts()
                    fig1, ax1 = plt.subplots(figsize=(10, 6))
                    track_counts.plot(kind='bar', color='skyblue', ax=ax1)
                    plt.xlabel('Track')
                    plt.ylabel('Number of Participants')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig1)
                
                with col2:
                    # Chart 2: Participation by Day
                    st.markdown('<h3 style="color: #0D47A1;">📅 Participation Trend by Day</h3>', unsafe_allow_html=True)
                    day_counts = filtered_df['Date'].value_counts().sort_index()
                    fig2, ax2 = plt.subplots(figsize=(10, 6))
                    day_counts.plot(kind='line', marker='o', color='green', ax=ax2)
                    plt.xlabel('Date')
                    plt.ylabel('Number of Participants')
                    plt.tight_layout()
                    st.pyplot(fig2)
            
            with tab2:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Chart 3: Geographic Distribution (by State)
                    st.markdown('<h3 style="color: #0D47A1;">🗺️ Participation by State</h3>', unsafe_allow_html=True)
                    state_counts = filtered_df['State'].value_counts().nlargest(10)
                    fig3, ax3 = plt.subplots(figsize=(10, 6))
                    state_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax3)
                    plt.axis('equal')
                    plt.tight_layout()
                    st.pyplot(fig3)
                
                with col2:
                    # Chart 4: College-wise Participation
                    st.markdown('<h3 style="color: #0D47A1;">🏫 Top Participating Colleges</h3>', unsafe_allow_html=True)
                    college_counts = filtered_df['College'].value_counts().nlargest(10)
                    fig4, ax4 = plt.subplots(figsize=(10, 6))
                    college_counts.plot(kind='barh', color='orange', ax=ax4)
                    plt.xlabel('Number of Participants')
                    plt.ylabel('College')
                    plt.tight_layout()
                    st.pyplot(fig4)
            
            with tab3:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Chart 5: Performance Metrics by Track
                    st.markdown('<h3 style="color: #0D47A1;">🎯 Performance Metrics by Track</h3>', unsafe_allow_html=True)
                    
                    score_metrics = filtered_df.groupby('Track')[
                        ['Technical_Score', 'Presentation_Score', 'Innovation_Score', 'Total_Score']
                    ].mean().round(2)
                    
                    fig5, ax5 = plt.subplots(figsize=(12, 8))
                    score_metrics[['Technical_Score', 'Presentation_Score', 'Innovation_Score']].plot(
                        kind='bar', ax=ax5
                    )
                    plt.xlabel('Track')
                    plt.ylabel('Average Score')
                    plt.title('Average Scores by Track')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig5)
                
                with col2:
                    # Chart 6: Heatmap of scores
                    st.markdown('<h3 style="color: #0D47A1;">🔥 Score Correlation Heatmap</h3>', unsafe_allow_html=True)
                    score_cols = ['Technical_Score', 'Presentation_Score', 'Innovation_Score', 'Total_Score']
                    corr = filtered_df[score_cols].corr()
                    fig6, ax6 = plt.subplots(figsize=(10, 8))
                    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax6)
                    plt.tight_layout()
                    st.pyplot(fig6)
                    
    elif menu == "Text Analysis":
        if 'data' not in st.session_state:
            st.warning("Please generate or upload a dataset from the Home page first!")
            return
            
        df = st.session_state.data
        
        st.markdown('<h2 class="sub-header">📝 Feedback Analysis</h2>', unsafe_allow_html=True)
        
        # Filter by track for text analysis with a nicer UI
        st.markdown('<div class="card">', unsafe_allow_html=True)
        track_filter = st.selectbox(
            "Select Track for Feedback Analysis",
            options=df['Track'].unique()
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        track_df = df[df['Track'] == track_filter]
        
        # Create tabs for different text analyses
        tab1, tab2, tab3 = st.tabs(["Word Cloud", "Sentiment Analysis", "Similarity Analysis"])
        
        with tab1:
            # Word Cloud
            st.markdown(f'<h3 style="color: #0D47A1;">🔤 Word Cloud for {track_filter}</h3>', unsafe_allow_html=True)
            
            # Combine all feedback for the selected track
            all_feedback = " ".join(track_df['Feedback'].tolist())
            
            # Generate word cloud
            wordcloud = WordCloud(
                width=800, height=400,
                background_color='white',
                max_words=100,
                contour_width=3,
                contour_color='steelblue'
            ).generate(all_feedback)
            
            # Display the generated image
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        
        with tab2:
            # Sentiment Distribution
            st.markdown(f'<h3 style="color: #0D47A1;">😊 Sentiment Analysis for {track_filter}</h3>', unsafe_allow_html=True)
            
            # Simple sentiment analysis based on total score
            track_df['Sentiment'] = pd.cut(
                track_df['Total_Score'],
                bins=[0, 70, 85, 100],
                labels=['Negative', 'Neutral', 'Positive']
            )
            
            sentiment_counts = track_df['Sentiment'].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(10, 6))
                sentiment_counts.plot(kind='bar', color=['red', 'yellow', 'green'], ax=ax)
                plt.xlabel('Sentiment')
                plt.ylabel('Count')
                plt.title(f'Sentiment Distribution for {track_filter}')
                st.pyplot(fig)
            
            with col2:
                # Pie chart for sentiment distribution
                fig, ax = plt.subplots(figsize=(10, 6))
                sentiment_counts.plot(kind='pie', autopct='%1.1f%%', colors=['red', 'yellow', 'green'], ax=ax)
                plt.axis('equal')
                plt.title(f'Sentiment Distribution for {track_filter}')
                st.pyplot(fig)
            
            # Display sample feedback by sentiment
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f'<h4 style="color: #0D47A1;">Sample Feedback by Sentiment</h4>', unsafe_allow_html=True)
            
            for sentiment in ['Positive', 'Neutral', 'Negative']:
                if sentiment in track_df['Sentiment'].values:
                    sample = track_df[track_df['Sentiment'] == sentiment].sample(min(3, sum(track_df['Sentiment'] == sentiment)))
                    st.markdown(f"<h5>{sentiment} Feedback:</h5>", unsafe_allow_html=True)
                    for _, row in sample.iterrows():
                        st.markdown(f"<div class='highlight'><i>\"{row['Feedback']}\"</i></div>", unsafe_allow_html=True)
                        st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab3:
            # Text Similarity Analysis
            st.markdown(f'<h3 style="color: #0D47A1;">📊 Feedback Similarity Analysis for {track_filter}</h3>', unsafe_allow_html=True)
            
            # Get unique feedback texts to avoid redundancy
            unique_feedback = track_df['Feedback'].unique()
            
            if len(unique_feedback) > 1:
                # Create TF-IDF vectors
                vectorizer = TfidfVectorizer(stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(unique_feedback)
                
                # Compute cosine similarity
                cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
                
                # Display similarity matrix
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(cosine_sim, annot=True, cmap='YlGnBu', ax=ax)
                plt.title(f'Feedback Similarity Matrix for {track_filter}')
                st.pyplot(fig)
                
                # Display most similar feedbacks
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("<h4 style='color: #0D47A1;'>Most Similar Feedback Pairs</h4>", unsafe_allow_html=True)
                
                # Get indices of most similar pairs (excluding self-comparisons)
                np.fill_diagonal(cosine_sim, 0)  # Zero out diagonal
                most_similar_indices = np.unravel_index(
                    np.argsort(cosine_sim.flatten())[-5:], cosine_sim.shape
                )
                
                for i, j in zip(most_similar_indices[0], most_similar_indices[1]):
                    if i < j:  # To avoid duplicates
                        st.markdown(f"<div class='highlight'>", unsafe_allow_html=True)
                        st.markdown(f"<b>Similarity Score: {cosine_sim[i][j]:.2f}</b>", unsafe_allow_html=True)
                        st.markdown(f"<i>Feedback 1:</i> {unique_feedback[i]}", unsafe_allow_html=True)
                        st.markdown(f"<i>Feedback 2:</i> {unique_feedback[j]}", unsafe_allow_html=True)
                        st.markdown("</div><br>", unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.write("Not enough unique feedback entries for similarity analysis.")
            
    elif menu == "Image Gallery":
        st.markdown('<h2 class="sub-header">📸 Event Photos Gallery</h2>', unsafe_allow_html=True)
        
        # Create tabs for sample images vs uploaded images
        gallery_tabs = st.tabs(["Sample Images", "Upload Your Images"])
        
        with gallery_tabs[0]:
            if 'images' not in st.session_state:
                st.warning("Please generate a dataset from the Home page first to create sample images!")
            else:
                # Day selection
                st.markdown('<div class="card">', unsafe_allow_html=True)
                day = st.selectbox(
                    "Select Day",
                    options=["Day1", "Day2", "Day3", "Day4"]
                )
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Display track images for selected day
                st.markdown(f'<h3 style="color: #0D47A1;">Photos from {day}</h3>', unsafe_allow_html=True)
                
                day_track_images = st.session_state.images[day]
                
                # Create columns to display images
                cols = st.columns(2)
                
                for idx, (track, img_path) in enumerate(day_track_images.items()):
                    with cols[idx % 2]:
                        track_name = track.replace('_', ' ')
                        st.markdown(f"<div class='card'>", unsafe_allow_html=True)
                        st.markdown(f"<h4 style='color: #0D47A1; background-color: #d1e4f5; padding: 8px; border-radius: 5px;'>{track_name}</h4>", unsafe_allow_html=True)
                        
                        # Read image
                        img = cv2.imread(img_path)
                        
                        # Image processing options
                        filter_options = ['original', 'grayscale', 'blur', 'edges', 'enhance', 'sharpen']
                        selected_filter = st.selectbox(
                            f"Choose filter for {track_name}",
                            options=filter_options,
                            key=f"{day}_{track}"
                        )
                        
                        # Apply selected filter
                        processed_img = apply_filter(img, selected_filter)
                        
                        # Improve the image display section
                        if len(processed_img.shape) == 2:  # Grayscale
                            processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)
                            st.image(processed_img_rgb, use_container_width=True, caption=f"{track_name} - {selected_filter}")
                        else:
                            # Ensure proper color conversion for display
                            img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                            st.image(img_rgb, use_container_width=True, caption=f"{track_name} - {selected_filter}")
                        
                        # Improve download button styling and image handling
                        if len(processed_img.shape) == 2:  # Grayscale
                            download_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)
                        else:
                            download_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                        
                        # Add styled download button
                        btn_style = "display: inline-block; padding: 8px 12px; background-color: #1E88E5; color: white; text-decoration: none; border-radius: 4px; margin-top: 10px;"
                        st.markdown(
                            get_image_download_link(
                                download_img, 
                                f"{day}_{track}_{selected_filter}.jpg",
                                f'<span style="{btn_style}">📥 Download Image</span>'
                            ),
                            unsafe_allow_html=True
                        )
                        st.markdown("</div>", unsafe_allow_html=True)
        
        with gallery_tabs[1]:
            st.markdown(f'<h3 style="color: #0D47A1;">Upload and Process Your Images</h3>', unsafe_allow_html=True)
            
            # Initialize session state for uploaded images if it doesn't exist
            if 'uploaded_images' not in st.session_state:
                st.session_state.uploaded_images = []
            
            # Image upload section
            st.markdown('<div class="card">', unsafe_allow_html=True)
            uploaded_files = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
            
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    if uploaded_file not in [img[0] for img in st.session_state.uploaded_images]:
                        # Read the image
                        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                        
                        # Add to session state
                        st.session_state.uploaded_images.append((uploaded_file, image))
                        
                st.success(f"{len(uploaded_files)} images uploaded successfully!")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Clear button
            if st.session_state.uploaded_images and st.button("Clear All Uploaded Images", key="clear_uploads"):
                st.session_state.uploaded_images = []
                st.success("All uploaded images cleared!")
                st.rerun()
            
            # Display and process uploaded images
            if st.session_state.uploaded_images:
                st.markdown('<h4 style="color: #0D47A1;">Process Your Uploaded Images</h4>', unsafe_allow_html=True)
                
                # Create a grid layout for images
                cols = st.columns(2)
                
                for idx, (file, img) in enumerate(st.session_state.uploaded_images):
                    with cols[idx % 2]:
                        # Create a card for each image
                        st.markdown(f"<div class='card'>", unsafe_allow_html=True)
                        st.markdown(f"<h4 style='color: #0D47A1; background-color: #d1e4f5; padding: 8px; border-radius: 5px;'>{file.name}</h4>", unsafe_allow_html=True)
                        
                        # Image processing options
                        filter_options = ['original', 'grayscale', 'blur', 'edges', 'enhance', 'sharpen']
                        selected_filter = st.selectbox(
                            f"Choose filter",
                            options=filter_options,
                            key=f"upload_{idx}"
                        )
                        
                        # Apply selected filter
                        processed_img = apply_filter(img, selected_filter)
                        
                        # Improve the image display section
                        if len(processed_img.shape) == 2:  # Grayscale
                            processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)
                            st.image(processed_img_rgb, use_container_width=True, caption=f"{file.name} - {selected_filter}")
                        else:
                            # Ensure proper color conversion for display
                            img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                            st.image(img_rgb, use_container_width=True, caption=f"{file.name} - {selected_filter}")
                        
                        # Improve download button styling and image handling
                        if len(processed_img.shape) == 2:  # Grayscale
                            download_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)
                        else:
                            download_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                        
                        # Add styled download button
                        btn_style = "display: inline-block; padding: 8px 12px; background-color: #1E88E5; color: white; text-decoration: none; border-radius: 4px; margin-top: 10px;"
                        st.markdown(
                            get_image_download_link(
                                download_img, 
                                f"{file.name.split('.')[0]}_{selected_filter}.jpg",
                                f'<span style="{btn_style}">📥 Download Image</span>'
                            ),
                            unsafe_allow_html=True
                        )
                        st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("Upload some images to start processing them!")

# Run the main application
if __name__ == "__main__":
    main()