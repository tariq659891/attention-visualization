import os
from pathlib import Path
import subprocess
from pydub import AudioSegment
from moviepy.editor import VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip

# Define the transcript segments with their timing (start_time in seconds)
# Adjusted to fit within 140 seconds (2:20) video duration
TRANSCRIPT_SEGMENTS = [
    {
        "start_time": 0.0,
        "text": "Welcome to this explanation of Multi-Head Attention, a key component in modern transformer architectures. Today we'll explore how it works and why it matters for large language models."
    },
    {
        "start_time": 8.0,
        "text": "Let's start with a simple sentence: 'I love deep learning.' Each word is a token in our sequence. Here, we have 4 tokens, giving us n equals 4."
    },
    {
        "start_time": 16.0,
        "text": "When these tokens enter the model, each one gets converted into a numerical vector called an embedding. For simplicity, we'll use embeddings with dimension d_model equals 8."
    },
    {
        "start_time": 24.0,
        "text": "Let's focus on the 'love' token. In attention mechanisms, we transform each token embedding into three different vectors:"
    },
    {
        "start_time": 30.0,
        "text": "A Query vector representing what this token is looking for, a Key vector representing what information this token contains, and a Value vector containing the actual information to be retrieved."
    },
    {
        "start_time": 39.0,
        "text": "For every token, we compute these Q, K, and V vectors using learned weight matrices. The dimensions d_k and d_v are typically set to 4 in our simplified example."
    },
    {
        "start_time": 48.0,
        "text": "Now, imagine we're trying to predict the next token after 'I love deep learning.' The model generates a Query vector for this position."
    },
    {
        "start_time": 55.0,
        "text": "The key insight of attention is that we compute similarity scores between this Query and all the Keys from previous tokens. We calculate dot products between the Query vector and each Key vector."
    },
    {
        "start_time": 64.0,
        "text": "In our example, the highest score is with the 'deep' token, showing that the model finds this most relevant for predicting the next word."
    },
    {
        "start_time": 70.0,
        "text": "We then apply softmax to convert these scores into probabilities. The 'deep' token gets the highest attention weight of 0.43, while others receive lower weights."
    },
    {
        "start_time": 78.0,
        "text": "Finally, we compute a weighted sum of all Value vectors, with these attention weights determining how much each token contributes. This creates a context vector that helps predict the next token - in this case, 'models' would be a likely prediction."
    },
    {
        "start_time": 88.0,
        "text": "Standard attention has a significant memory problem. For long sequences like 32,000 tokens, storing all Key and Value vectors requires enormous memory."
    },
    {
        "start_time": 96.0,
        "text": "This is where Multi-Latent Attention comes in. Instead of storing separate K and V vectors for all 32,000 tokens, MLA compresses them into just 512 latent vectors."
    },
    {
        "start_time": 105.0,
        "text": "The benefits are substantial: Queries attend to 512 vectors instead of 32,000, memory usage is reduced by over 98%, inference speed is much faster, and much longer context windows become practical."
    },
    {
        "start_time": 116.0,
        "text": "Let's recap what we've learned: Q, K, V are the fundamental components of attention. The number of tokens is represented by n. The dimensions of keys and values are d_k and d_v."
    },
    {
        "start_time": 126.0,
        "text": "Queries match with Keys to find relevant context. The context vector is a weighted sum of Value vectors. MLA compresses Keys and Values to save memory."
    },
    {
        "start_time": 134.0,
        "text": "Multi-Latent Attention enables efficient inference with long context windows, making today's powerful large language models practical. Thanks for watching!"
    }
]

from gtts import gTTS

def generate_audio_with_tts(transcript_segments, output_dir="audio_segments"):
    """Generate audio files using Google Text-to-Speech"""
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    audio_files = []
    
    for i, segment in enumerate(transcript_segments):
        # File path for the segment (mp3 for gTTS)
        file_path = os.path.join(output_dir, f"segment_{i:02d}.mp3")
        audio_files.append(file_path)
        
        # Use Google Text-to-Speech
        tts = gTTS(text=segment["text"], lang='en', slow=False)
        tts.save(file_path)
        
        print(f"Generated audio segment {i+1}/{len(transcript_segments)}")
    
    return audio_files

def combine_audio_segments(audio_files, output_file="full_narration.mp3", video_duration_ms=141700):
    """Combine separate audio files into a single audio track
    
    Args:
        audio_files: List of audio file paths
        output_file: Path to save the combined audio
        video_duration_ms: Duration of the video in milliseconds (default: 141700 = 2:21.7)
    """
    combined = AudioSegment.empty()
    
    # Expected timings for segments (in milliseconds)
    expected_starts = [segment["start_time"] * 1000 for segment in TRANSCRIPT_SEGMENTS]
    current_pos = 0
    
    # Track total duration to ensure we don't exceed video length
    total_audio_duration = 0
    
    print(f"Video duration: {video_duration_ms/1000:.2f} seconds")
    print(f"Processing {len(audio_files)} audio segments...")
    
    for i, (file_path, expected_start) in enumerate(zip(audio_files, expected_starts)):
        # Load the audio segment
        segment = AudioSegment.from_file(file_path)
        segment_duration = len(segment)
        
        # Calculate how much silence we need before this segment
        expected_pos = expected_start
        silence_needed = max(0, expected_pos - current_pos)
        
        # Check if adding this segment would exceed video duration
        if current_pos + silence_needed + segment_duration > video_duration_ms:
            # If we're about to exceed video duration, trim the segment
            available_time = max(0, video_duration_ms - current_pos - silence_needed)
            if available_time > 0:
                segment = segment[:available_time]
                print(f"⚠️ Trimmed segment {i+1} to fit within video duration ({available_time/1000:.2f}s)")
            else:
                print(f"⚠️ Skipping segment {i+1} as it would exceed video duration")
                continue
        
        # Add silence if needed
        if silence_needed > 0:
            combined += AudioSegment.silent(duration=silence_needed)
            current_pos += silence_needed
            print(f"Added {silence_needed/1000:.2f}s of silence before segment {i+1}")
        
        # Add the segment
        combined += segment
        current_pos += len(segment)
        total_audio_duration += len(segment) + silence_needed
        
        print(f"Added segment {i+1} at position {current_pos/1000:.2f}s, duration: {len(segment)/1000:.2f}s")
        
        # Stop if we've reached the video duration
        if current_pos >= video_duration_ms:
            print(f"Reached video duration limit after segment {i+1}")
            break
    
    # Add final silence if needed to match video duration
    if current_pos < video_duration_ms:
        final_silence = video_duration_ms - current_pos
        combined += AudioSegment.silent(duration=final_silence)
        print(f"Added {final_silence/1000:.2f}s of silence at the end to match video duration")
    
    # Export the combined audio
    combined.export(output_file, format="mp3")
    print(f"Exported combined audio to {output_file} (total duration: {len(combined)/1000:.2f}s)")
    
    return output_file

def overlay_audio_on_video(video_path, audio_path, output_path):
    """Overlay the narration audio on the video"""
    # Load the video
    video = VideoFileClip(video_path)
    
    # Load the audio
    audio = AudioFileClip(audio_path)
    
    # Set the audio of the video
    video = video.set_audio(audio)
    
    # Write the result
    video.write_videofile(output_path)
    print(f"Exported final video to {output_path}")

def add_subtitles_to_video(video_path, transcript_segments, output_path):
    """Add subtitles to the video based on transcript segments"""
    video = VideoFileClip(video_path)
    video_duration = video.duration
    print(f"Video duration: {video_duration:.2f} seconds")
    
    subtitle_clips = []
    
    for i, segment in enumerate(transcript_segments):
        # Create text clip for subtitle
        txt_clip = TextClip(
            segment["text"], 
            fontsize=24, 
            color='white',
            bg_color='black',
            stroke_color='black',
            stroke_width=1,
            method='caption',
            size=(video.size[0] - 80, None)  # Width slightly less than video for better readability
        )
        
        # Position at bottom of frame with padding
        txt_clip = txt_clip.set_position(('center', 'bottom')).margin(bottom=40, opacity=0)
        
        # Calculate duration based on next segment's start time or remaining video time
        if i < len(transcript_segments) - 1:
            next_start = transcript_segments[i + 1]["start_time"]
            duration = next_start - segment["start_time"]
        else:
            # For the last segment, use the remaining video duration
            duration = video_duration - segment["start_time"]
            # Cap at a reasonable maximum if needed
            duration = min(duration, 10.0)
        
        # Ensure subtitle doesn't extend beyond video duration
        end_time = min(segment["start_time"] + duration, video_duration)
        actual_duration = end_time - segment["start_time"]
        
        print(f"Subtitle {i+1}: Start={segment['start_time']:.1f}s, Duration={actual_duration:.1f}s, End={end_time:.1f}s")
        
        # Set timing with calculated duration
        txt_clip = txt_clip.set_start(segment["start_time"]).set_duration(actual_duration)
        
        subtitle_clips.append(txt_clip)
    
    # Create final video with subtitles
    final_video = CompositeVideoClip([video] + subtitle_clips)
    
    # Write output (preserve original audio)
    final_video.write_videofile(output_path, audio_codec='aac')
    print(f"Exported video with subtitles to {output_path}")

def main():
    # File paths
    video_path = "/media/tariq/NewVolume/LLM/animation/KV cache/attention-visualization/media/videos/multi_head_attention/1080p60/MultiHeadAttentionScene.mp4"  # Your rendered animation
    output_dir = "audio_segments"
    combined_audio = "full_narration.mp3"
    output_video = "multi_head_attention_with_narration.mp4"
    output_with_subtitles = "multi_head_attention_with_narration_and_subtitles.mp4"
    
    # Get video duration
    try:
        video_clip = VideoFileClip(video_path)
        video_duration = video_clip.duration
        video_duration_ms = int(video_duration * 1000)
        print(f"\nDetected video duration: {video_duration:.2f} seconds ({video_duration_ms} ms)")
        video_clip.close()
    except Exception as e:
        print(f"Error getting video duration: {e}")
        print("Using default duration of 141.7 seconds")
        video_duration_ms = 141700  # Default to 2:21.7
    
    print("\n1. Generating audio segments...")
    # Generate audio segments
    audio_files = generate_audio_with_tts(TRANSCRIPT_SEGMENTS, output_dir)
    
    print("\n2. Combining audio segments...")
    # Combine audio segments with correct video duration
    narration_audio = combine_audio_segments(audio_files, combined_audio, video_duration_ms)
    
    print("\n3. Overlaying audio on video...")
    # Overlay audio on video
    overlay_audio_on_video(video_path, narration_audio, output_video)
    
    print("\n4. Adding subtitles...")
    # Add subtitles
    add_subtitles_to_video(output_video, TRANSCRIPT_SEGMENTS, output_with_subtitles)
    
    print("\nProcess completed successfully!")
    print(f"Final video with narration and subtitles: {output_with_subtitles}")

if __name__ == "__main__":
    main()