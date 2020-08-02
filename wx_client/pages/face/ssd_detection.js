//index.js
//获取应用实例
var time = null;
var myCanvas = null;
var windowHeight, windowWidth;
var type = null;
Page({
  data: {
    device: true,
    camera: true,
    x1: 0,
    y1: 0,
    x2: 0,
    y2: 0,
  },
  onLoad() {
    this.setData({
      ctx: wx.createCameraContext(),
      device: this.data.device,
    })
    wx.getSystemInfo({
      success: function (res) {
        console.log(res);
        // 屏幕宽度、高度
        windowHeight = res.windowHeight;
        windowWidth = res.windowWidth;
        console.log('height=' + res.windowHeight);
        console.log('width=' + res.windowWidth);
      }
    })
  },
  open() {
    this.setData({
      camera: true
    })
    type = "takePhoto";
    let ctx = wx.createCameraContext(this)
    let that = this
    time = setInterval(function () {
      if (type == "takePhoto") {
        console.log("begin takephoto")
        ctx.takePhoto({
          quality: "normal",
          success: (res) => {
            console.log(res.tempImagePath)
            var tempImagePath = res.tempImagePath
            wx.uploadFile({
              url: 'http://127.0.0.1:90/upload',
              filePath: tempImagePath,
              name: 'file',
              header: { "Content-type": "multipart/form-data" },
              success: function (res) {
                var im_path = res.data
                console.log(im_path)
                wx.request({
                  url: 'http://127.0.0.1:90/face_detect?url=' + im_path,
                  method: "GET",
                  header: { "Content-type": "application/json" },
                  success: function (res) {
                    var pos = res.data
                    pos[0] = parseInt(pos[0] * windowWidth)
                    pos[1] = parseInt(pos[1] * windowHeight * 0.6)
                    pos[2] = parseInt(pos[2] * windowWidth)
                    pos[3] = parseInt(pos[3] * windowHeight * 0.6)
                    that.setData({
                      x1: pos[0],
                      y1: pos[1],
                      x2: pos[2],
                      y2: pos[3],
                    })

                    myCanvas = wx.createCanvasContext("myCanvas")
                    myCanvas.drawImage(tempImagePath, 0, 0, windowWidth, windowHeight * 0.6)
                    myCanvas.setStrokeStyle("red")
                    myCanvas.setLineWidth(5)
                    myCanvas.rect(pos[0], pos[1], (pos[2] - pos[0]), (pos[3] - pos[1]))
                    myCanvas.stroke()
                    myCanvas.draw()


                  }
                })
              }
            })
          }
        })
      }
    }, 1000)
  },
  // 关闭模拟的相机界面
  close() {
    console.log("关闭相机");
    type = "endPhoto"
  },
})
