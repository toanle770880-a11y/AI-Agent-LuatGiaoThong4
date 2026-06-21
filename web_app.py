__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import time
from google import genai
import PIL.Image
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ----------------- 1. LOAD SYSTEM -----------------
@st.cache_resource
def load_ai_system():
    # ✅ LẤY API KEY TỪ ENV
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except:
        st.error("❌ Chưa thiết lập GOOGLE_API_KEY trong Secrets")
        st.stop()

    client = genai.Client(api_key=api_key)

    # Vector DB
    embed_model = HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    vector_db = Chroma(
        persist_directory="./my_vector_db",
        embedding_function=embed_model
    )
    # Tăng k lên 7 để tra cứu sâu hơn
    retriever = vector_db.as_retriever(search_kwargs={"k": 15})

    return client, retriever

client, retriever = load_ai_system()

# ----------------- 2. HÀM GỌI AI TỔNG HỢP (HỖ TRỢ CẢ ẢNH + TEXT) -----------------
def call_gemini_smart(prompt, image=None, model="gemini-3.5-flash"):
    """Hàm gọi AI có cơ chế thử lại và hỗ trợ đa phương thức"""
    for i in range(3): # Thử lại 3 lần nếu lỗi
        try:
            contents = [prompt]
            if image:
                contents.append(image)
            
            res = client.models.generate_content(
                model=model,
                contents=contents
            )
            if res and res.text:
                return res.text
        except Exception as e:
            print(f"Lần thử {i+1} thất bại: {e}")
            time.sleep(2)
    return "⚠️ AI đang bận hoặc có lỗi kết nối, thử lại phát nữa nhé!"

# ----------------- 3. DATA QUIZ (GIỮ NGUYÊN CỦA ÔNG) -----------------
quiz_data = [
    {
        "question": "Câu 1: Người điều khiển xe mô tô hai bánh có được phép sử dụng ô (dù) khi đang lái xe không?",
        "options": ["A. Có, nhưng chỉ khi trời mưa nhỏ.", "B. Không, tuyệt đối cấm.", "C. Có, chỉ dành cho người ngồi sau.", "D. Tùy thuộc vào quy định của từng địa phương."],
        "answer": "B",
        "explanation": "Theo Khoản 3 Điều 30 Luật Giao thông đường bộ 2008, cấm người lái xe mô tô hai bánh sử dụng ô."
    },
    {
        "question": "Câu 2: Thứ tự các xe đi như thế nào là đúng quy tắc giao thông?",
        "image": "images/sahinh1.png", 
        "options": ["A. Xe tải, xe khách, xe con, mô tô.", "B. Xe tải, mô tô, xe khách, xe con.", "C. Xe khách, xe tải, xe con, mô tô.", "D. Mô tô, xe khách, xe tải, xe con."],
        "answer": "B",
        "explanation": "💡 **Phân tích sa hình:** Xe tải và Mô tô đi trước do nằm trên đường ưu tiên."
    },
    {
        "question": "Câu 3: Thứ tự các xe đi như thế nào là đúng quy tắc giao thông?",
        "image": "images/sahinh2.png", 
        "options": [
            "A. Xe công an, xe con, xe tải, xe khách.", 
            "B. Xe công an, xe khách, xe con, xe tải.", 
            "C. Xe công an, xe tải, xe khách, xe con.", 
            "D. Xe con, xe công an, xe tải, xe khách."
        ],
        "answer": "A",
        "explanation": "💡 **Phân tích sa hình:**\n\n1. **Xe ưu tiên:** Xe công an đang làm nhiệm vụ nên được quyền đi đầu tiên.\n2. **Đường ưu tiên:** Xe tải (và xe con) nằm trên đường ưu tiên (biển báo tam giác hướng lên). Xe khách nằm trên đường không ưu tiên (biển tam giác ngược) nên phải nhường đường, đi cuối cùng.\n3. **Hướng không vướng:** Lúc này xét xe con và xe tải. Khi xe công an đã đi khỏi, ngã tư bên phải của xe con là đường trống (không vướng). Trong khi đó bên phải của xe tải đang vướng xe con. Vậy xe con đi trước xe tải.\n\n✅ **Thứ tự đúng:** Xe công an -> Xe con -> Xe tải -> Xe khách."
    },
    {
        "question": "Câu 4: Theo hướng mũi tên, thứ tự các xe đi như thế nào là đúng quy tắc giao thông?",
        "image": "images/sahinh3.png", 
        "options": [
            "A. Xe tải, xe công an, xe khách, xe con.", 
            "B. Xe công an, xe khách, xe con, xe tải.", 
            "C. Xe công an, xe con, xe tải, xe khách.", 
            "D. Xe công an, xe tải, xe khách, xe con."
        ],
        "answer": "D",
        "explanation": "💡 **Phân tích sa hình:**\n\n1. **Xe ưu tiên:** Xe công an làm nhiệm vụ đi đầu tiên.\n2. **Đường ưu tiên:** Xe tải nằm trên đường ưu tiên (biển báo hình thoi) nên được đi thứ hai.\n3. **Hướng đi:** Xe khách và xe con nằm trên đường không ưu tiên (biển tam giác ngược). Áp dụng quy tắc hướng đi: Đi thẳng đi trước, rẽ trái nhường đường. Vậy xe khách đi thẳng được đi trước xe con rẽ trái.\n\n✅ **Thứ tự đúng:** Xe công an -> Xe tải -> Xe khách -> Xe con."
    },
    {
        "question": "Câu 5: Thứ tự các xe đi như thế nào là đúng quy tắc giao thông?",
        "image": "images/sahinh4.png", 
        "options": [
            "A. Xe tải, xe con, mô tô.", 
            "B. Xe con, xe tải, mô tô.", 
            "C. Mô tô, xe con, xe tải.", 
            "D. Xe con, mô tô, xe tải."
        ],
        "answer": "C",
        "explanation": "💡 **Phân tích sa hình:**\n\n1. **Đường đồng cấp:** Không có xe ưu tiên, không có biển báo đường ưu tiên.\n2. **Quyền bên phải không vướng:** \n- Xét tại ngã tư, nhánh đường bên dưới đang trống.\n- Suy ra, phía bên tay phải của **Mô tô** không có xe $\\rightarrow$ Mô tô được đi đầu tiên.\n- Sau khi Mô tô đi, bên phải của **Xe con** trống $\\rightarrow$ Xe con đi thứ 2.\n- Cuối cùng là **Xe tải**.\n\n✅ **Thứ tự đúng:** Mô tô -> Xe con -> Xe tải."
    },
    {
        "question": "Câu 6: Xe nào phải nhường đường trong trường hợp này?",
        "image": "images/sahinh5.png", 
        "options": [
            "A. Xe con.", 
            "B. Xe tải."
        ],
        "answer": "A",
        "explanation": "💡 **Phân tích sa hình:**\n\n1. **Quy tắc vòng xuyến:** Tại ngã tư giao nhau có đặt biển báo hiệu đi theo vòng xuyến, người lái xe phải nhường đường cho xe đi đến từ bên **trái**.\n2. **Áp dụng:** Xe tải đang ở trong vòng xuyến (đến từ bên trái của xe con). Xe con đang ở ngoài chuẩn bị tiến vào vòng xuyến nên bắt buộc phải nhường đường.\n\n✅ **Kết luận:** Xe con phải nhường đường."
    },
    {
        "question": "Câu 7: Trường hợp này xe nào được quyền đi trước?",
        "image": "images/sahinh6.png", 
        "options": [
            "A. Mô tô.", 
            "B. Xe con."
        ],
        "answer": "B",
        "explanation": "💡 **Phân tích sa hình:**\n\n1. **Biển báo hiệu:** Quan sát thấy phía trước mặt người điều khiển mô tô có biển báo hiệu lệnh **STOP** (Dừng lại).\n2. **Quy tắc:** Theo Luật Giao thông đường bộ, khi gặp biển báo STOP, tất cả các phương tiện đều phải dừng lại và nhường đường cho xe đi trên đường ưu tiên hoặc xe đi từ các hướng khác tới.\n\n✅ **Kết luận:** Xe mô tô phải dừng lại nhường đường, **Xe con** được quyền đi trước."
    },
    {
        "question": "Câu 8: Thứ tự các xe đi như thế nào là đúng quy tắc giao thông?",
        "image": "images/sahinh7.png", 
        "options": [
            "A. Xe con (A), xe cứu thương, xe con (B).", 
            "B. Xe cứu thương, xe con (B), xe con (A).", 
            "C. Xe con (B), xe con (A), xe cứu thương."
        ],
        "answer": "A",
        "explanation": "💡 **Phân tích sa hình:**\n\n1. **Nhất chớm (Xe đã vào giao lộ):** Quan sát kỹ sẽ thấy **Xe con (A)** đã đi qua vạch dừng người đi bộ và tiến hẳn vào trong ngã tư. Theo luật, xe đã vào giao lộ được quyền đi trước tiên, **kể cả khi có xe ưu tiên** đi tới.\n2. **Nhì ưu (Xe ưu tiên):** Sau khi xe con (A) đi, quyền ưu tiên tiếp theo thuộc về **xe cứu thương**.\n3. **Cuối cùng:** Xe con (B) đi cuối.\n\n✅ **Thứ tự đúng:** Xe con (A) -> Xe cứu thương -> Xe con (B)."
    },
    {
        "question": "Câu 9: Thứ tự các xe đi như thế nào là đúng quy tắc giao thông?",
        "image": "images/sahinh8.png", 
        "options": [
            "A. Xe cứu thương, xe cứu hỏa, xe con.", 
            "B. Xe cứu hỏa, xe cứu thương, xe con.", 
            "C. Xe cứu thương, xe con, xe cứu hỏa."
        ],
        "answer": "B",
        "explanation": "💡 **Phân tích sa hình:**\n\n1. **Quy tắc xe ưu tiên:** Theo Luật Giao thông đường bộ, thứ tự xe ưu tiên được quy định như sau: Xe chữa cháy (cứu hỏa) $\\rightarrow$ Xe quân sự, xe công an $\\rightarrow$ Xe cứu thương.\n2. **Áp dụng:** Trong hình có 2 xe ưu tiên, đối chiếu theo luật thì xe cứu hỏa có quyền ưu tiên cao nhất, sau đó mới đến xe cứu thương. Xe con là xe bình thường đi cuối cùng.\n\n✅ **Thứ tự đúng:** Xe cứu hỏa -> Xe cứu thương -> Xe con."
    },
    {
        "question": "Câu 10: Xe nào được quyền đi trước trong trường hợp này?",
        "image": "images/sahinh9.png", 
        "options": [
            "A. Xe mô tô.", 
            "B. Xe cứu thương.", 
            "C. Xe con.", 
            "D. Xe tải."
        ],
        "answer": "B",
        "explanation": "💡 **Phân tích sa hình:**\n\n1. **Xe ưu tiên:** Trong hình có xe cứu thương đang phát tín hiệu ưu tiên đi làm nhiệm vụ.\n2. **Quy tắc:** Theo Luật Giao thông đường bộ, xe ưu tiên (hỏa táng, quân sự, công an, cứu thương) được quyền đi trước các xe khác khi qua nơi giao nhau, bất kể hướng đi hay biển báo đường ưu tiên.\n3. **Kết luận:** Dù xe mô tô đang ở trên đường ưu tiên, nhưng vẫn phải nhường đường cho **Xe cứu thương** đi trước.\n\n✅ **Đáp án đúng:** B. Xe cứu thương."
    },
    # (Ông tự copy nốt các câu quiz còn lại của ông vào đây nhé)
]

# ----------------- 4. UI -----------------
st.set_page_config(page_title="Trợ lý Giao thông AI", page_icon="🚦")

# Chèn logo từ thư mục images vào Sidebar
# Lưu ý: Nhớ đổi tên file ảnh ICTU thành logo_ictu.png và bỏ vào thư mục images
st.sidebar.image("images/logo_ictu.png", width=200) 

st.sidebar.markdown("<h3 style='text-align: center;'>ICTU - Đại học Công nghệ Thông tin & Truyền thông</h3>", unsafe_allow_html=True)
st.sidebar.divider()

st.title("🚦 Trợ lý Luật Giao thông Việt Nam")
# Thêm dòng này vào ngay trên dòng "with tab1:"
tab1, tab2 = st.tabs(["💬 Hỏi đáp", "📝 Trắc nghiệm"])

# ================= TAB 1: HỎI ĐÁP =================
with tab1:
    st.header("💬 Chat & Nhận diện hình ảnh")
    
    # --- LOGIC XÓA ẢNH SAU KHI HỎI ---
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    # Dùng key động để có thể reset uploader
    uploaded_file = st.file_uploader(
        "🖼️ Đính kèm ảnh (biển báo, tình huống...)", 
        type=["jpg", "jpeg", "png"],
        key=f"uploader_{st.session_state.uploader_key}"
    )
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Ảnh đang chọn", width=250)
    
    with st.form(key='chat_form', clear_on_submit=True):
        user_input = st.text_input("Nhập câu hỏi tại đây:", placeholder="VD: Biển báo này có ý năng gì?...")
        submit_button = st.form_submit_button(label='Gửi câu hỏi 🚀')

    # 2. Xử lý Logic khi nhấn nút Gửi
    if submit_button and user_input:
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        with st.spinner("Đang phân tích..."):
            current_image = None
            if uploaded_file is not None:
                import PIL.Image
                current_image = PIL.Image.open(uploaded_file)
                # TỐI ƯU PROMPT CHO VISION
                visual_prompt = (
                    f"Bạn là chuyên gia hàng đầu về Hệ thống biển báo hiệu giao thông đường bộ Việt Nam (QCVN 41:2019/BGTVT). "
                    f"Hãy quan sát thật kỹ bức ảnh và trả lời câu hỏi: '{user_input}'.\n\n"
                    f"ĐỂ ĐẢM BẢO TÍNH CHÍNH XÁC TUYỆT ĐỐI, BẠN HÃY PHÂN TÍCH THEO CÁC BƯỚC SAU:\n"
                    f"1. Trích xuất đặc điểm trực quan:\n"
                    f"   - Hình dáng biển báo (Tròn, tam giác đều, chữ nhật, hình thoi...).\n"
                    f"   - Màu sắc (Nền màu gì? Viền màu gì?).\n"
                    f"   - Biểu tượng bên trong (Có hình vẽ, con số, vạch kẻ hay chữ viết gì? Mô tả thật chi tiết).\n"
                    f"2. Xác định nhóm biển báo:\n"
                    f"   - Nếu tròn, viền đỏ: Thuộc nhóm Biển Cấm.\n"
                    f"   - Nếu tam giác đều, viền đỏ, nền vàng: Thuộc nhóm Biển Nguy Hiểm.\n"
                    f"   - Nếu tròn, nền xanh lam: Thuộc nhóm Biển Hiệu Lệnh.\n"
                    f"   - Nếu vuông/chữ nhật, nền xanh lam: Thuộc nhóm Biển Chỉ Dẫn.\n"
                    f"3. Định danh chính thức: Kết hợp Bước 1 và Bước 2, đối chiếu với kho kiến thứccủa bạn để gọi đúng MÃ SỐ BIỂN (VD: P.102, W.207a...) và TÊN GỌI của biển báo trong ảnh.\n"
                    f"4. Kết luận ý nghĩa: Giải thích ngắn gọn luật giao thông mà người điều khiển phương tiện phải tuân thủ khi gặp biển này.\n\n"
                    f"Lưu ý quan trọng: Suy luận từng bước. Nếu hình ảnh mờ không thể đọc chính xác, hãy yêu cầu người dùng cung cấp ảnh rõ hơn, tuyệt đối không bịa ra tên biển báo."
                )
                answer = call_gemini_smart(visual_prompt, image=current_image)
            else:
                docs = retriever.invoke(user_input)
                context = "\n".join([doc.page_content for doc in docs])
                
                # TỐI ƯU PROMPT ĐỂ AI TRÍCH DẪN ĐIỀU LUẬT (CHIÊU 3 + CĂN CỨ LUẬT)
                rag_prompt = (
                    f"DỮ LIỆU LUẬT TRÍCH XUẤT: {context}\n"
                    f"CÂU HỎI CỦA NGƯỜI DÙNG: {user_input}\n\n"
                    f"YÊU CẦU QUAN TRỌNG:\n"
                    f"1. Xác định đúng đối tượng: Nếu người dùng không nói rõ, hãy mặc định trả lời cho XE MÔ TÔ (XE MÁY).\n"
                    f"2. Truy xuất chính xác: Tìm trong dữ liệu đoạn nào dành riêng cho phương tiện đó. "
                    f"Ví dụ: Điều 7 dành cho Mô tô, Điều 5 dành cho Ô tô, Điều 11 dành cho xe thô sơ/vật nuôi.\n"
                    f"3. Cấu trúc câu trả lời:\n"
                    f"   - Phương tiện áp dụng: (Ví dụ: Xe mô tô, xe gắn máy).\n"
                    f"   - Căn cứ pháp lý: (Nêu rõ Điều, Khoản).\n"
                    f"   - Mức xử phạt: (Lấy đúng con số tương ứng với phương tiện đó).\n"
                    f"   - Lời khuyên: ....\n"
                    f"LƯU Ý: Tuyệt đối không lấy mức phạt của xe thô sơ áp dụng cho xe máy."
                )
                answer = call_gemini_smart(rag_prompt)

        # LƯU VÀO LỊCH SỬ CHAT: Lưu cả câu hỏi và ảnh vào cùng một tin nhắn của User
        st.session_state.messages.append({
            "role": "user", 
            "content": user_input, 
            "image": current_image # Lưu ảnh vào đây để tí nữa hiển thị lại
        })

        # Lưu tin nhắn của Assistant
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer
        })

        # --- CHIÊU CUỐI: TỰ XÓA ẢNH Ở Ô NHẬP LIỆU ---
        if uploaded_file is not None:
            st.session_state.uploader_key += 1 # Tăng key để reset widget uploader
        
        st.rerun()

    # 3. Hiển thị tin nhắn
    if "messages" in st.session_state:
        for m in reversed(st.session_state.messages):
            with st.chat_message(m["role"]):
                # Nếu tin nhắn có ảnh (User gửi), hiện ảnh lên trước
                if "image" in m and m["image"] is not None:
                    st.image(m["image"], width=250)
                st.markdown(m["content"])
# ================= TAB 2: TRẮC NGHIỆM =================
with tab2:
    st.header("📝 Trắc nghiệm")
    if 'quiz_index' not in st.session_state:
        st.session_state.quiz_index, st.session_state.score, st.session_state.submitted = 0, 0, False

    idx = st.session_state.quiz_index
    if idx < len(quiz_data):
        item = quiz_data[idx]
        st.subheader(item["question"])
        if "image" in item: st.image(item["image"], use_container_width=True)
        
        choice = st.radio("Chọn đáp án:", item["options"], index=None, key=f"q_{idx}")
        if st.button("Nộp bài"):
            if choice:
                st.session_state.submitted = True
                if choice[0] == item["answer"]:
                    st.success("✅ Chính xác!")
                    st.session_state.score += 1
                else:
                    st.error(f"❌ Sai rồi! Đáp án đúng là {item['answer']}")
                with st.expander("Xem giải thích"):
                    st.markdown(item["explanation"])
            else: st.warning("Hãy chọn một đáp án!")

        if st.session_state.submitted and st.button("Câu tiếp theo ➡️"):
            st.session_state.quiz_index += 1
            st.session_state.submitted = False
            st.rerun()
    else:
        st.balloons()
        st.success(f"Chúc mừng! Điểm của bạn: {st.session_state.score}/{len(quiz_data)}")
        if st.button("Làm lại từ đầu"):
            st.session_state.quiz_index = 0
            st.session_state.score = 0
            st.rerun()
