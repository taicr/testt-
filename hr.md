```mermaid
graph TD
    subgraph "Giao diện & Người dùng"
        User(Nhà Tuyển Dụng) -- "1. /create-jd" --> RC(Rocket.Chat)
    end

    subgraph "Hệ thống Tuyển dụng (ATS)"
        ATS(Hệ thống ATS)
    end

    subgraph "Bộ não AI (LangGraph Supervisor)"
        Supervisor((Router/Orchestrator))
        State[(Trạng tháiJob ID, Status, CV list...)]

        %% Tools
        GenJD(Tool: Tạo JD)
        UploadJD(Tool: Tải JD lên ATS)
        FetchCVs(Tool: Lấy CV từ ATS)
        AnalyzeCVs(Tool: Phân tích & Chấm điểm CV)
        SendEmail(Tool: Gửi Email)
    end
    
    %% Luồng đi
    RC -- "2. Webhook kích hoạt" --> Supervisor

    Supervisor -- "3. Gọi Tool Tạo JD" --> GenJD
    GenJD -- "4. Trả về JD thô" --> Supervisor

    Supervisor -- "5. Gửi JD & Nút bấm về RC" --> RC
    Supervisor -- "PAUSE nCập nhật State: CHỜ PHÊ DUYỆT" --> State

    RC -- "6. User bấm 'Phê duyệt'" --> Supervisor

    subgraph "Supervisor Ra Quyết Định #1"
        direction LR
        Supervisor_Decide1{"Phản hồi từ User?"}
    end
    Supervisor --> Supervisor_Decide1
    
    Supervisor_Decide1 -- "Phê duyệt" --> UploadJD
    Supervisor_Decide1 -- "Yêu cầu sửa" --> GenJD

    UploadJD -- "7. Gọi API của ATS" --> ATS
    UploadJD -- "8. Báo cáo thành công" --> Supervisor
    
    Supervisor -- "PAUSE Cập nhật State: 'ĐANG CHỜ CV'\n(Chờ trigger định kỳ)" --> State
    
    subgraph "Tác vụ định kỳ (Cron Job)"
        Scheduler("Mỗi 24h") -- "9. Kích hoạt Supervisor" --> Supervisor
    end

    Supervisor -- "10. Gọi Tool Lấy CV" --> FetchCVs
    FetchCVs -- "11. Lấy CV từ ATS" --> ATS
    FetchCVs -- "12. Trả về danh sách CV" --> Supervisor

    subgraph "Supervisor Ra Quyết Định #2"
        direction LR
        Supervisor_Decide2{"Có CV mới không?"}
    end
    Supervisor --> Supervisor_Decide2

    Supervisor_Decide2 -- "Có" --> AnalyzeCVs
    Supervisor_Decide2 -- "Không" --> State

    AnalyzeCVs -- "13. Trả về CV đã chấm điểm" --> Supervisor

    subgraph "Supervisor Ra Quyết Định #3"
        direction LR
        Supervisor_Decide3{"Điểm CV thế nào?"}
    end
    Supervisor --> Supervisor_Decide3

    Supervisor_Decide3 -- "Điểm > 85(Mời phỏng vấn)" --> SendEmail
    Supervisor_Decide3 -- "Điểm < 60(Từ chối)" --> SendEmail
    Supervisor_Decide3 -- "Khác(Lưu lại)" --> State
    
    SendEmail -- "14. Báo cáo hoàn thành" --> Supervisor
    Supervisor -- "END Kết thúc một nhánh xử lý" --> State
