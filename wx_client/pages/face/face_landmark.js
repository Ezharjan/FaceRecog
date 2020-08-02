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
      camera: true,
      //login_res: "定位中..."
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
              url: 'http://127.0.0.1:90/face_landmark',
              filePath: tempImagePath,
              name: 'file',
              header: { "Content-type": "multipart/form-data" },
              success: function (res) {
                if (res.data == "error") {
                }
                else{
                  console.log(res.data);
                  var pos = res.data.split(",");
                  myCanvas = wx.createCanvasContext("myCanvas");
                  myCanvas.drawImage(tempImagePath, 0, 0, windowWidth, windowHeight * 0.6);
                  myCanvas.setStrokeStyle("green");
                  myCanvas.setLineWidth(5);
                  for(var i=0;i<136 * 2;){
                    var x = parseInt(pos[i+2] * windowWidth);
                    var y = parseInt(pos[i + 3] * windowHeight * 0.6);
                    myCanvas.moveTo(x,y);
                    myCanvas.lineTo(x+1, y+1);
                    i += 4;
                  }
 
                  var diff1 = pos[67 * 4 + 1] - pos[63 * 4 + 1];  
                  var diff2 = pos[41 * 4 + 1] - pos[37 * 4 + 1];
                  var diff3 = pos[46 * 4 + 1] - pos[44 * 4 + 1];
                  console.log(diff1, diff2, diff3)
//准确率，检出率
//召回率
                  if(diff2  < 0.03 && diff3 < 0.03){
                    that.setData({
                      eye_flag:"闭眼"
                    })
                  } 
                  else{
                    that.setData({
                      eye_flag: "睁眼"
                    })
                  }

                  if(diff1 > 0.02){
                    that.setData({
                      mouth_flag:"张嘴"
                    })
                  }
                  else{
                    that.setData({
                      mouth_flag: "闭嘴"
                    })
                  }

                  myCanvas.stroke()
                  myCanvas.draw()

                }

              },
            })
          }
        })
      }
    }, 500)
  },
  // 关闭模拟的相机界面
  close() {
    console.log("关闭相机");
    type = "endPhoto"
    this.setData({
      camera: true,
      //login_res: "停止"
    })
  },
})
