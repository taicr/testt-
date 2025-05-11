```mermaid
graph TD
    A[Hệ thống WEB TRANH] --> B[Quản lý Người dùng];
    A --> C[Quản lý Sản phẩm];
    A --> D[Quản lý Đơn hàng];
    A --> E[Chức năng Mua sắm];
    A --> F[Chức năng Tìm kiếm];
    A --> G[Hỗ trợ Khách hàng];

    B --> B1[Đăng ký];
    B --> B2[Đăng nhập];
    B --> B3[Quản lý Tài khoản Cá nhân];
    B --> B4[Quản lý Danh sách Người dùng - Quan tri vien];

    C --> C1[Xem Danh sách Sản phẩm];
    C --> C2[Xem Chi tiết Sản phẩm];
    C --> C3[Quản lý Sản phẩm - Quan tri vien];

    D --> D1[Tạo Đơn hàng];
    D --> D2[Xem Lịch sử Đơn hàng];
    D --> D3[Quản lý Đơn hàng - Quan tri vien];
    D --> D4[Cập nhật Trạng thái Đơn hàng];

    E --> E1[Thêm vào Giỏ hàng];
    E --> E2[Quản lý Giỏ hàng];
    E --> E3[Thanh toán khi nhan hang, ma Q R];
    E --> E4[Quản lý Địa chỉ Giao hàng];
    E --> E5[Đánh giá Sản phẩm];

    F --> F1[Tìm kiếm Cơ bản];
    F --> F2[Tìm kiếm Nâng cao RAG ];

    G --> G1[Chatbot Hỗ trợ];
