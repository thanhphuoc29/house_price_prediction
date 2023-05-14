
function run() {
    var btn = document.getElementById('predict');
    btn.addEventListener('click', sendData);
}
run();

function sendData() {
    document.getElementById('myForm').addEventListener('submit', (event) => {
        event.preventDefault();
    });
    var square = document.querySelector('input[name="area"]').value;
    var bedrooms = document.querySelector('input[name="bedrooms"]').value;
    var bathrooms = document.querySelector('input[name="bathrooms"]').value;
    var pos = document.querySelector('#mySelect').value;
    let arr = pos.split(",");
    let longtitude = parseInt(arr[0]);
    let latitude = parseInt(arr[1])

    var data = {
        area: square,
        Bedroom: bedrooms,
        Bathroom: bathrooms,
        longtitude: longtitude,
        latitude: latitude
    };
    var xhttp = new XMLHttpRequest();
    xhttp.onload = function () {
        var respone = xhttp.responseText;
        var data = JSON.parse(respone);
        console.log(data)
        document.getElementById('price_predict').innerHTML = data[0].price_predict
        // document.getElementById('price').innerHTML = data.price
        // document.getElementById('num_bed').innerHTML = data.num_bath
        // document.getElementById('num_bath').innerHTML = data['num_bath']
        // document.getElementById('area_room').innerHTML = data.area
        // document.getElementById('pos').innerHTML = data['pos']
        var showInfor = document.getElementById("showInfor");

        let html = '';

        for (let i = 0; i < data.length; i++) {
            let home = data[i];
            html += `
                <div class="home">
                <h3>Thông tin ngôi nhà ${i + 1}</h3>
                <ul>
                    <li>Số phòng ngủ: ${home.num_room}</li>
                    <li>Số phòng tắm: ${home.num_bath}</li>
                    <li>Diện tích: ${home.area} m2</li>
                    <li>Mức giá: ${home.price}</li>
                    <li>Vị trí: ${home.pos}</li>
                </ul>
                </div>
            `;
        }

        document.getElementById('showInfor').innerHTML = html;

        const resultContainer = document.querySelector('.result-container');

        // Thêm sự kiện nghe click vào nút dự đoán
        document.getElementById('myForm').addEventListener('submit', (event) => {
            event.preventDefault(); // Ngăn chặn gửi form mặc định

            // Hiển thị phân tích kết quả dự đoán với hiệu ứng fade-in và slide-in
            resultContainer.classList.add('show');
            resultContainer.classList.remove('hide');
            console.log(respone)
        })
    }
    var api = "http://127.0.0.1:5000/api";
    xhttp.open("POST", api, true);
    xhttp.setRequestHeader("Content-type", "application/json");
    console.log(data)
    xhttp.send(JSON.stringify(data));
    // console.log(square + " " + bedrooms + " " + bathrooms + " " + year_built + " " + room);
    // Ngăn chặn gửi form mặc định khi nhấn nút "Submit"
}