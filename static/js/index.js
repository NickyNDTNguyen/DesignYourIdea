(function() {
	var canvas = document.querySelector("#canvas");
	var context = canvas.getContext("2d");
	canvas.width = 600;
	canvas.height = 600;

	var Mouse = {x:0, y:0};
	var lastMouse = {x:0, y:0};
	context.fillStyle = "white";
	context.fillRect(0, 0, canvas.width, canvas.height);
	context.color = "black";
	context.lineWidth = 2;
    context.lineJoin = context.lineCap = 'round';
	
	debug();


	var history = {
		redo_list: [],
		undo_list: [],
		saveState: function(canvas, list, keep_redo) {
		  keep_redo = keep_redo || false;
		  if(!keep_redo) {
			this.redo_list = [];
		  }
		  
		  this.undo_list.push(canvas.toDataURL());   
		},
		undo: function(canvas, context) {
		  this.restoreState(canvas, context, this.undo_list, this.redo_list);
		},
		redo: function(canvas, context) {
		  this.restoreState(canvas, context, this.redo_list, this.undo_list);
		},
		restoreState: function(canvas, context,  pop, push) {
			if (pop.length) {
                this.saveState(canvas, push, true);
                var img = document.createElement("img");
                img.setAttribute('src', this.undo_list[this.undo_list.length-2]);
				this.undo_list = this.undo_list.slice(0,this.undo_list.length-2)
                img.setAttribute('alt', 'canvas');
                img.onload = function () {
			  context.clearRect(0, 0, 600, 600);
			  context.drawImage(img, 0, 0, 600, 600, 0, 0, 600, 600);  
			}
		  }
		}
	  }




	canvas.addEventListener("mousemove", function(e) {
		lastMouse.x = Mouse.x;
		lastMouse.y = Mouse.y;

		// Mouse.x = e.pageX - this.offsetLeft-15;
		// Mouse.y = e.pageY - this.offsetTop-15;
		Mouse.x = e.pageX - this.offsetLeft-5;
		Mouse.y = e.pageY - this.offsetTop-5;
		// history.saveState(canvas, [],true)

	}, false);

	canvas.addEventListener("mousedown", function(e) {
		canvas.addEventListener("mousemove", onPaint, false);
		history.saveState(canvas, [],true)
	}, false);

	canvas.addEventListener("mouseup", function() {
		canvas.removeEventListener("mousemove", onPaint, false);

	}, false);

	    
    document.getElementById('undo').addEventListener('click', function() {
		// console.log(history.undo_list[history.undo_list.length-1])
		history.undo(canvas, context);
	  });



	var onPaint = function() {	
		context.lineWidth = context.lineWidth;
		context.lineJoin = "round";
		context.lineCap = "round";
		context.strokeStyle = context.color;
	
		context.beginPath();
		context.moveTo(lastMouse.x, lastMouse.y);
		context.lineTo(Mouse.x,Mouse.y );
		context.closePath();
		context.stroke();
	};


	function debug() {
		$("#clearButton").on("click", function() {
			context.clearRect( 0, 0, 280, 280 );
			context.fillStyle="white";
			context.fillRect(0,0,canvas.width,canvas.height);
		});
	}

	
}());