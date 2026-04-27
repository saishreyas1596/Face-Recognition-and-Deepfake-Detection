import streamlit as st
import numpy as np
import cv2
from PIL import Image
from utils.predict import *

st.set_page_config(page_title="Face Recognition System", layout="wide")

# Initialize session state
if 'total' not in st.session_state:
    st.session_state.total = 0
    st.session_state.real = 0
    st.session_state.fake = 0
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'faces_data' not in st.session_state:
    st.session_state.faces_data = []
if 'result_image' not in st.session_state:
    st.session_state.result_image = None
if 'last_uploaded_file' not in st.session_state:
    st.session_state.last_uploaded_file = None

st.title("👤 Face Recognition & Deepfake Detection")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📊 Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Faces", st.session_state.total)
    with col2:
        st.metric("✅ Real", st.session_state.real)
    
    col3, col4 = st.columns(2)
    with col3:
        st.metric("❌ Fake", st.session_state.fake)
    with col4:
        accuracy = 0
        if st.session_state.total > 0:
            accuracy = (st.session_state.real / st.session_state.total) * 100
        st.metric("Accuracy", f"{accuracy:.1f}%")
    
    st.markdown("---")
    st.header("📚 Saved People")
    
    all_faces = get_all_faces()
    if all_faces:
        for person in all_faces:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"👤 {person}")
            with col2:
                if st.button(f"🗑️", key=f"del_{person}"):
                    delete_face(person)
                    st.rerun()
    else:
        st.info("No faces saved yet")

# Main content
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg"])

# Clear results when new file is uploaded
if uploaded_file and uploaded_file != st.session_state.last_uploaded_file:
    st.session_state.analysis_done = False
    st.session_state.faces_data = []
    st.session_state.result_image = None
    st.session_state.last_uploaded_file = uploaded_file

if uploaded_file:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original", use_column_width=True)
    
    if st.button("🔍 Analyze Image", use_container_width=True):
        with st.spinner("Analyzing..."):
            faces, boxes = detect_face(img_array)
            
            if not faces:
                st.error("No faces detected!")
                st.session_state.analysis_done = False
            else:
                st.success(f"Found {len(faces)} face(s)")
                
                result_img = img_array.copy()
                faces_list = []
                
                for i, (face, (x1, y1, x2, y2)) in enumerate(zip(faces, boxes)):
                    # Recognize
                    person, conf = recognize_face(face)
                    
                    # Deepfake
                    df_result, df_conf = predict_deepfake(face)
                    
                    if df_result == "REAL":
                        color = (0, 255, 0)
                        status = "✅ REAL"
                        st.session_state.real += 1
                    else:
                        color = (0, 0, 255)
                        status = "❌ FAKE"
                        st.session_state.fake += 1
                    st.session_state.total += 1
                    
                    # Draw on image
                    cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 3)
                    label = f"{person}"
                    cv2.putText(result_img, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Store face data
                    faces_list.append({
                        'index': i,
                        'face': face,
                        'person': person,
                        'confidence': conf,
                        'deepfake_result': df_result,
                        'deepfake_confidence': df_conf,
                        'status': status
                    })
                
                st.session_state.faces_data = faces_list
                st.session_state.result_image = result_img
                st.session_state.analysis_done = True
    
    # Display results if analysis is done
    if st.session_state.analysis_done and st.session_state.faces_data:
        for face_info in st.session_state.faces_data:
            with st.container():
                st.markdown(f"### Face {face_info['index']+1}")
                
                col_a, col_b, col_c = st.columns([1, 1.5, 1.5])
                
                with col_a:
                    st.image(face_info['face'], width=150)
                
                with col_b:
                    if face_info['person'] != "Unknown":
                        st.success(f"**Recognized as:** {face_info['person']}")
                        st.write(f"**Confidence:** {face_info['confidence']:.1%}")
                    else:
                        st.warning(f"**Recognized as:** Unknown Person")
                    
                    if face_info['deepfake_result'] == "REAL":
                        st.success(f"**Deepfake:** {face_info['status']} ({face_info['deepfake_confidence']:.0%})")
                    else:
                        st.error(f"**Deepfake:** {face_info['status']} ({face_info['deepfake_confidence']:.0%})")
                
                with col_c:
                    # Save form
                    with st.form(key=f"save_form_{face_info['index']}"):
                        st.write("**Add to Database:**")
                        default_name = "" if face_info['person'] == "Unknown" else face_info['person']
                        name_input = st.text_input(
                            "Name:", 
                            value=default_name,
                            key=f"name_input_{face_info['index']}",
                            placeholder="Enter person's name"
                        )
                        submitted = st.form_submit_button("💾 Save This Face")
                        
                        if submitted and name_input:
                            success, count = add_new_face(name_input.strip(), face_info['face'])
                            if success:
                                st.success(f"✅ '{name_input}' saved! (Total {count} images for this person)")
                                st.balloons()
                                st.rerun()
                            else:
                                st.error("Failed to save!")
                        elif submitted and not name_input:
                            st.warning("Please enter a name!")
                
                st.markdown("---")
        
        # Display result image
        if st.session_state.result_image is not None:
            with col2:
                st.image(st.session_state.result_image, caption="Analyzed Image", use_column_width=True)

# Footer
st.markdown("---")
st.markdown("""
<p style="text-align: center; color: gray;">
    💡 <strong>For best accuracy:</strong> Save multiple photos of the same person (3-5 images)
</p>
""", unsafe_allow_html=True)