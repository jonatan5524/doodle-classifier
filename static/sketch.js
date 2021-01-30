let points;
let loading;
let start;
let sendInterval;
let lastPredict;

function setup() {
  createCanvas(400, 400);
  strokeWeight(3);
  frameRate(30);

  points = [];

  start = false;
  loading = true;
  fetch("/load")
    .then((res) => {
      if (!res.ok) {
        throw Error(res.text);
      }

      loading = false;
    })
    .catch((err) => {
      console.error(err);
    });
}

function draw() {
  if (loading) {
    background(220);
  } else {
    const WHITE = color(255, 255, 255);
    const BLACK = color(0, 0, 0);

    background(WHITE);

    points.forEach(({ x, y }) => {
      fill(BLACK);
      ellipse(x, y, 10);
    });
  }
}

function mouseDragged() {
  if (mouseIsPressed) {
    points.push({
      x: mouseX,
      y: mouseY,
    });
  }
}

function prepareImage() {
  const canvas = document.getElementsByTagName("canvas")[0];
  const ctx = canvas.getContext("2d");

  const imageData = ctx.getImageData(0, 0, 400, 400);
  const tfImage = tf.browser.fromPixels(imageData, 1);

  //Resize to 28X28
  let tfResizedImage = tf.image.resizeBilinear(tfImage, [28, 28]);
  //Since white is 255 black is 0 so need to revert the values
  //so that white is 0 and black is 255
  tfResizedImage = tf.cast(tfResizedImage, "float32");
  tfResizedImage = tf
    .abs(tfResizedImage.sub(tf.scalar(255)))
    .div(tf.scalar(255))
    .flatten();
  return tfResizedImage.reshape([1, 784]);
}

function sendPredict() {
  const data = Array.from(prepareImage().dataSync());

  fetch("/capture", {
    method: "POST",
    headers: {
      "Content-Type": "image/png",
      "Access-Control-Allow-Origin": URL,
    },
    body: data,
  })
    .then((res) => {
      if (!res.ok) {
        throw Error(res.text);
      }

      return res.text();
    })
    .then((prediction) => {
      if (lastPredict != undefined) {
        document.getElementById(lastPredict).style.color = "Black";
      }
      lastPredict = prediction;

      document.getElementById(prediction).style.color = "Green";
      const msg = new SpeechSynthesisUtterance(prediction);
      window.speechSynthesis.speak(msg);
    })
    .catch((err) => {
      console.error(err);
    });
}

function mousePressed() {
  if (!start) {
    sendInterval = setInterval(sendPredict, 3000);

    start = true;
  }

  points.push({
    x: mouseX,
    y: mouseY,
  });
}

function keyPressed() {
  if (keyCode === 32) {
    points = [];
    start = false;

    if (sendInterval != undefined) {
      clearInterval(sendInterval);
    }
    if (lastPredict != undefined) {
      document.getElementById(lastPredict).style.color = "Black";
    }
  }
}
