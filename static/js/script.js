(function($) {
    var tool;
   
    var canvas = document.querySelector("#canvas");
    var context = canvas.getContext('2d');
    
    var history = {
      redo_list: [],
      undo_list: [],
      saveState: function(canvas, list, keep_redo) {
        keep_redo = keep_redo || false;
        if(!keep_redo) {
          this.redo_list = [];
        }
        
        (list || this.undo_list).push(canvas.toDataURL());   
      },
      undo: function(canvas, context) {
        this.restoreState(canvas, context, this.undo_list, this.redo_list);
      },
      redo: function(canvas, context) {
        this.restoreState(canvas, context, this.redo_list, this.undo_list);
      },
      restoreState: function(canvas, context,  pop, push) {
        if(pop.length) {
          this.saveState(canvas, push, true);
          var restore_state = pop.pop();
          var img = new Element('img', {'src':restore_state});
          img.onload = function() {
            context.clearRect(0, 0, 600, 400);
            context.drawImage(img, 0, 0, 600, 400, 0, 0, 600, 400);  
          }
        }
      }
    }
    
    var pencil = {
      options: {
        stroke_color: ['00', '00', '00'],
        dim: 4
      },
      init: function(canvas, context) {
        this.canvas = canvas;
        this.canvas_coords = this.canvas.getCoordinates();
        this.context = context;
        this.context.strokeColor = this.options.stroke_color;
        this.drawing = false;
        this.addCanvasEvents();
      },
      addCanvasEvents: function() {
        this.canvas.addEvent('mousedown', this.start.bind(this));
        this.canvas.addEvent('mousemove', this.stroke.bind(this));
        this.canvas.addEvent('mouseup', this.stop.bind(this));
        this.canvas.addEvent('mouseout', this.stop.bind(this));
      },
      start: function(evt) {
        var x = evt.page.x - this.canvas_coords.left;
        var y = evt.page.y - this.canvas_coords.top;
        this.context.beginPath();
        this.context.moveTo(x, y);
        history.saveState(this.canvas);
        this.drawing = true;
      },
      stroke: function(evt) {
        if(this.drawing) {
          var x = evt.page.x - this.canvas_coords.left;
          var y = evt.page.y - this.canvas_coords.top;
          this.context.lineTo(x, y);
          this.context.stroke();
          
        }
      },
      stop: function(evt) {
        if(this.drawing) this.drawing = false;
      }
    };
    
    $('pencil').addEvent('click', function() {
      pencil.init(canvas, context);
    });
    
    $('undo').addEvent('click', function() {
      history.undo(canvas, context);
    });
    
    $('redo').addEvent('click', function() {
      history.redo(canvas, context);
    });
  
    
  })(document.id)