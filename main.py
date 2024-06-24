import predict

# res = predict.response("Nào mở cửa?")
# print(res)


from flask import Flask, request, jsonify
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/chatbot', methods=['POST'])
def chatbot():
      # Nhận tin nhắn từ trang web ReactJS dưới dạng chuỗi JSON
    request_data = request.json
    message = request_data['message']

    print('message: ', message)

    # Xử lý tin nhắn bằng chatbot và nhận phản hồi
    response = predict.response(message)
    # res_str = json.dumps(response)
    
    print("Phản hồi từ chatbot:", type(response))

    # Chuyển đổi phản hồi từ string sang JSON để trả về cho trang web ReactJS
    response_data = {'response': response}
    # print(jsonify(response_data))
    # Trả lại phản hồi cho trang web ReactJS dưới dạng chuỗi JSON
    return jsonify(response_data)



    #   # Nhận tin nhắn từ trang web ReactJS
    # message = request.json['message']

    # # Xử lý tin nhắn bằng chatbot Python
    # response = predict.response(message)  # Thay thế your_chatbot_function bằng hàm xử lý tin nhắn của chatbot

    # # Chuyển đổi phản hồi từ string thành JSON object
    # response_json = json.dumps(response)

    # # Trả lại phản hồi cho trang web ReactJS
    # return jsonify({'response': response_json})


if __name__ == '__main__':
    app.run(debug=True, port=5000)  # Chạy ứng dụng Flask trong chế độ debug
