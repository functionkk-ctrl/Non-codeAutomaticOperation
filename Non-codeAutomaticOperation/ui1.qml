import QtQuick 2.15
import QtQuick.Controls 2.15
import PySide6

ApplicationWindow {
    visible: true
    width: 480
    height: 360
    color: "transparent"

    
    // 顯示一個簡單的文本
    Text {
        anchors.centerIn: parent
        text: "Hello, World!"
        color: "white"
        font.pixelSize: 20
    }
}
