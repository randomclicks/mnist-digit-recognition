import pygame
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
from PIL import Image
import time

class DrawingApp:
    def __init__(self, model_path='models/mnist_model.h5'):
        # Initialize pygame
        pygame.init()
        
        # Define themes
        self.themes = {
            'light': {
                'bg': (245, 245, 245),
                'canvas_bg': (10, 10, 10),
                'sidebar_bg': (230, 230, 230),
                'text': (40, 40, 40),
                'accent': (0, 120, 212),
                'accent_hover': (0, 90, 182),
                'button_bg': (255, 255, 255),
                'button_hover': (235, 235, 235),
                'success': (0, 170, 0),
                'error': (220, 50, 50),
                'grid': (50, 50, 50, 30),
                'border': (200, 200, 200)
            },
            'dark': {
                'bg': (35, 35, 35),
                'canvas_bg': (10, 10, 10),
                'sidebar_bg': (45, 45, 45),
                'text': (230, 230, 230),
                'accent': (0, 140, 255),
                'accent_hover': (20, 160, 255),
                'button_bg': (60, 60, 60),
                'button_hover': (80, 80, 80),
                'success': (0, 190, 0),
                'error': (255, 70, 70),
                'grid': (180, 180, 180, 30),
                'border': (80, 80, 80)
            }
        }
        
        # Set initial theme
        self.current_theme = 'light'
        self.colors = self.themes[self.current_theme]
        
        # Set up drawing surface
        self.canvas_width, self.canvas_height = 500, 500
        self.sidebar_width = 300
        self.screen_width = self.canvas_width + self.sidebar_width
        self.screen_height = self.canvas_height
        
        # Create the main window
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption('Digit Recognizer - Enhanced UI')
        
        # Setup fonts
        self.title_font = pygame.font.SysFont('Arial', 26, bold=True)
        self.font = pygame.font.SysFont('Arial', 18)
        self.small_font = pygame.font.SysFont('Arial', 16)
        
        # Drawing properties
        self.drawing = False
        self.last_pos = None
        self.draw_color = (255, 255, 255)  # Default: white
        self.radius = 15  # Default brush size
        self.min_radius = 2
        self.max_radius = 30
        
        # Grid settings
        self.show_grid = False
        self.grid_size = 50
        
        # Prediction history
        self.prediction_history = []
        self.max_history = 5
        
        # Current prediction
        self.current_prediction = None
        self.current_confidence = None
        self.processed_img = None
        self.all_confidences = None
        
        # Button styling
        self.button_radius = 8
        
        # UI Layout
        sidebar_x = self.canvas_width + 25
        section_spacing = 30
        
        # Section 1: Title and Status (0-80px)
        title_y = 20
        status_y = title_y + 35
        
        # Section 2: Main Actions (100-200px)
        btn_width = 120
        btn_height = 40
        btn_spacing = 15
        main_actions_y = status_y + section_spacing
        
        # Create button layout
        self.buttons = {
            'clear': pygame.Rect(sidebar_x, main_actions_y, btn_width, btn_height),
            'predict': pygame.Rect(sidebar_x + btn_width + btn_spacing, main_actions_y, btn_width, btn_height),
            'save': pygame.Rect(sidebar_x, main_actions_y + btn_height + btn_spacing, btn_width, btn_height),
            'theme': pygame.Rect(sidebar_x + btn_width + btn_spacing, main_actions_y + btn_height + btn_spacing, btn_width, btn_height),
        }
        
        # Section 3: Drawing Tools (220-280px)
        tools_y = main_actions_y + (btn_height + btn_spacing) * 2 + section_spacing
        self.buttons.update({
            'brush_minus': pygame.Rect(sidebar_x, tools_y, 40, 40),
            'brush_plus': pygame.Rect(sidebar_x + 50, tools_y, 40, 40),
            'grid_toggle': pygame.Rect(sidebar_x + btn_width + btn_spacing, tools_y, btn_width, 40),
        })
        
        # Section 4: Prediction Results (300-500px)
        self.prediction_section_y = tools_y + 60
        
        # Button animations
        self.button_animations = {}
        
        # Animation variables
        self.last_action_time = 0
        self.action_message = None
        self.action_color = (255, 255, 255)
        self.action_duration = 2.0
        
        # Initialize drawing canvas
        self.clear_canvas()
        
        # Load model
        try:
            self.model = load_model(model_path)
            print("Model loaded successfully!")
            self.model_loaded = True
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Make sure to train the model first by running mnist_digit_recognition.py")
            self.model = None
            self.model_loaded = False
    
    def clear_canvas(self):
        """Create a fresh canvas for drawing."""
        self.canvas = pygame.Surface((self.canvas_width, self.canvas_height))
        self.canvas.fill(self.colors['canvas_bg'])
    
    def draw_button(self, rect, text, color=None, hover=False):
        """Draw a button with text."""
        if color is None:
            color = self.colors['button_bg']
        
        # Draw button background with rounded corners
        pygame.draw.rect(self.screen, 
                        self.colors['button_hover'] if hover else color,
                        rect, border_radius=self.button_radius)
        
        # Draw button text
        text_surface = self.font.render(text, True, self.colors['text'])
        text_rect = text_surface.get_rect(center=rect.center)
        self.screen.blit(text_surface, text_rect)
    
    def draw_prediction_results(self):
        """Draw the prediction results section."""
        y_pos = self.prediction_section_y
        
        # Draw section title with background
        title_height = 30
        title_rect = pygame.Rect(self.canvas_width + 15, y_pos - 5, self.sidebar_width - 30, title_height)
        pygame.draw.rect(self.screen, self.colors['button_bg'], title_rect, border_radius=self.button_radius)
        
        section_title = self.font.render("Prediction Results", True, self.colors['text'])
        self.screen.blit(section_title, (self.canvas_width + 25, y_pos))
        y_pos += title_height + 10
        
        if self.current_prediction is not None:
            # Create a background panel for results
            panel_rect = pygame.Rect(self.canvas_width + 15, y_pos - 5, 
                                   self.sidebar_width - 30, 180)
            pygame.draw.rect(self.screen, self.colors['button_bg'], 
                           panel_rect, border_radius=self.button_radius)
            
            # Draw main prediction result
            result_text = self.font.render(f"Predicted Digit: {self.current_prediction}", 
                                         True, self.colors['text'])
            self.screen.blit(result_text, (self.canvas_width + 25, y_pos))
            y_pos += 25
            
            confidence_text = self.font.render(f"Confidence: {self.current_confidence:.1f}%", 
                                             True, self.colors['text'])
            self.screen.blit(confidence_text, (self.canvas_width + 25, y_pos))
            y_pos += 35
            
            # Draw confidence bars
            if self.all_confidences is not None:
                bar_width = 180
                bar_height = 12
                bar_spacing = 16
                
                # Draw confidence bars in a panel
                for digit in range(10):
                    # Draw digit label
                    digit_text = self.small_font.render(f"{digit}:", True, self.colors['text'])
                    self.screen.blit(digit_text, (self.canvas_width + 25, y_pos))
                    
                    # Draw confidence bar background
                    bar_bg_rect = pygame.Rect(self.canvas_width + 45, y_pos + 2, 
                                            bar_width, bar_height)
                    pygame.draw.rect(self.screen, self.colors['border'], bar_bg_rect)
                    
                    # Draw confidence bar fill
                    confidence = self.all_confidences[digit]
                    bar_fill_width = int((confidence / 100) * bar_width)
                    bar_fill_rect = pygame.Rect(self.canvas_width + 45, y_pos + 2, 
                                              bar_fill_width, bar_height)
                    
                    # Color based on confidence
                    if digit == self.current_prediction:
                        bar_color = self.colors['accent']
                    elif confidence > 20:
                        bar_color = self.colors['success']
                    else:
                        bar_color = (100, 100, 100)
                    
                    pygame.draw.rect(self.screen, bar_color, bar_fill_rect)
                    
                    # Draw confidence percentage
                    conf_text = self.small_font.render(f"{confidence:.1f}%", 
                                                     True, self.colors['text'])
                    self.screen.blit(conf_text, (self.canvas_width + 45 + bar_width + 5, y_pos))
                    
                    y_pos += bar_spacing
        
        # Draw keyboard shortcuts at the bottom
        self.draw_keyboard_shortcuts()
    
    def draw_keyboard_shortcuts(self):
        """Draw keyboard shortcuts in a clean panel at the bottom."""
        shortcuts_y = self.screen_height - 120
        
        # Create background panel for shortcuts
        panel_height = 110
        panel_rect = pygame.Rect(self.canvas_width + 15, shortcuts_y - 5, 
                               self.sidebar_width - 30, panel_height)
        pygame.draw.rect(self.screen, self.colors['button_bg'], 
                        panel_rect, border_radius=self.button_radius)
        
        # Draw shortcuts
        shortcuts = [
            ("Keyboard Shortcuts:", True),
            ("C - Clear canvas", False),
            ("P - Predict digit", False),
            ("S - Save drawing", False),
            ("G - Toggle grid", False),
            ("T - Toggle theme", False),
            ("+/- - Adjust brush", False)
        ]
        
        for i, (text, is_title) in enumerate(shortcuts):
            shortcut_text = self.small_font.render(text, True, self.colors['text'])
            x_pos = self.canvas_width + (25 if is_title else 35)
            self.screen.blit(shortcut_text, (x_pos, shortcuts_y + i*15))
    
    def draw_buttons(self):
        """Draw all buttons."""
        # Draw main action buttons
        self.draw_button(self.buttons['clear'], "Clear", hover=self.buttons['clear'].collidepoint(pygame.mouse.get_pos()))
        self.draw_button(self.buttons['predict'], "Predict", hover=self.buttons['predict'].collidepoint(pygame.mouse.get_pos()))
        self.draw_button(self.buttons['save'], "Save", hover=self.buttons['save'].collidepoint(pygame.mouse.get_pos()))
        self.draw_button(self.buttons['theme'], "Theme", hover=self.buttons['theme'].collidepoint(pygame.mouse.get_pos()))
        
        # Draw brush size controls
        self.draw_button(self.buttons['brush_minus'], "-", hover=self.buttons['brush_minus'].collidepoint(pygame.mouse.get_pos()))
        self.draw_button(self.buttons['brush_plus'], "+", hover=self.buttons['brush_plus'].collidepoint(pygame.mouse.get_pos()))
        
        # Draw grid toggle button
        grid_text = "Grid: " + ("On" if self.show_grid else "Off")
        self.draw_button(self.buttons['grid_toggle'], grid_text, 
                        self.colors['accent'] if self.show_grid else None,
                        hover=self.buttons['grid_toggle'].collidepoint(pygame.mouse.get_pos()))
    
    def draw_grid(self):
        """Draw grid on the canvas."""
        if self.show_grid:
            grid_surface = pygame.Surface((self.canvas_width, self.canvas_height), pygame.SRCALPHA)
            for x in range(0, self.canvas_width, self.grid_size):
                pygame.draw.line(grid_surface, self.colors['grid'], (x, 0), (x, self.canvas_height))
            for y in range(0, self.canvas_height, self.grid_size):
                pygame.draw.line(grid_surface, self.colors['grid'], (0, y), (self.canvas_width, y))
            self.screen.blit(grid_surface, (0, 0))
    
    def draw(self):
        """Draw everything."""
        # Fill background
        self.screen.fill(self.colors['bg'])
        
        # Draw canvas background
        pygame.draw.rect(self.screen, self.colors['canvas_bg'], 
                        (0, 0, self.canvas_width, self.canvas_height))
        
        # Draw the canvas content
        self.screen.blit(self.canvas, (0, 0))
        
        # Draw grid if enabled
        self.draw_grid()
        
        # Draw sidebar background
        pygame.draw.rect(self.screen, self.colors['sidebar_bg'],
                        (self.canvas_width, 0, self.sidebar_width, self.screen_height))
        
        # Draw title
        title = self.title_font.render("Digit Recognizer", True, self.colors['text'])
        self.screen.blit(title, (self.canvas_width + 25, 20))
        
        # Draw brush size indicator
        brush_text = self.font.render(f"Brush Size: {self.radius}", True, self.colors['text'])
        self.screen.blit(brush_text, (self.canvas_width + 25, 60))
        
        # Draw buttons
        self.draw_buttons()
        
        # Draw prediction result
        self.draw_prediction_results()
        
        # Draw action message if available
        if self.action_message and time.time() - self.last_action_time < self.action_duration:
            # Fade out effect
            alpha = 255 * (1 - (time.time() - self.last_action_time) / self.action_duration)
            action_text = self.font.render(self.action_message, True, self.action_color)
            self.screen.blit(action_text, (self.canvas_width + 25, 80))
        
        # Update display
        pygame.display.flip()
    
    def handle_mouse_input(self, event):
        """Handle mouse input events."""
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Left click
            if event.button == 1:
                # Check if click is within canvas bounds
                if event.pos[0] < self.canvas_width:
                    self.drawing = True
                    self.last_pos = event.pos
                else:
                    # Check button clicks
                    mouse_pos = event.pos
                    if self.buttons['clear'].collidepoint(mouse_pos):
                        self.clear_canvas()
                        self.show_action_message("Canvas cleared")
                    elif self.buttons['predict'].collidepoint(mouse_pos):
                        self.predict()
                    elif self.buttons['save'].collidepoint(mouse_pos):
                        self.save_drawing()
                    elif self.buttons['theme'].collidepoint(mouse_pos):
                        self.toggle_theme()
                    elif self.buttons['brush_minus'].collidepoint(mouse_pos):
                        self.adjust_brush_size(-2)
                    elif self.buttons['brush_plus'].collidepoint(mouse_pos):
                        self.adjust_brush_size(2)
                    elif self.buttons['grid_toggle'].collidepoint(mouse_pos):
                        self.show_grid = not self.show_grid
                        self.show_action_message(f"Grid {'enabled' if self.show_grid else 'disabled'}")
        
        elif event.type == pygame.MOUSEBUTTONUP:
            # Left click release
            if event.button == 1:
                self.drawing = False
                self.last_pos = None
        
        elif event.type == pygame.MOUSEMOTION:
            # Draw if mouse is pressed and within canvas bounds
            if self.drawing and event.pos[0] < self.canvas_width:
                if self.last_pos:
                    pygame.draw.line(self.canvas, self.draw_color,
                                   self.last_pos, event.pos, self.radius * 2)
                pygame.draw.circle(self.canvas, self.draw_color,
                                 event.pos, self.radius)
                self.last_pos = event.pos
    
    def handle_keyboard_input(self, event):
        """Handle keyboard input events."""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:
                self.clear_canvas()
                self.show_action_message("Canvas cleared")
            elif event.key == pygame.K_p:
                self.predict()
            elif event.key == pygame.K_s:
                self.save_drawing()
            elif event.key == pygame.K_g:
                self.show_grid = not self.show_grid
                self.show_action_message(f"Grid {'enabled' if self.show_grid else 'disabled'}")
            elif event.key == pygame.K_t:
                self.toggle_theme()
            elif event.key in [pygame.K_PLUS, pygame.K_KP_PLUS, pygame.K_EQUALS]:
                self.adjust_brush_size(2)
            elif event.key in [pygame.K_MINUS, pygame.K_KP_MINUS]:
                self.adjust_brush_size(-2)
    
    def adjust_brush_size(self, delta):
        """Adjust the brush size."""
        new_radius = max(self.min_radius, min(self.max_radius, self.radius + delta))
        if new_radius != self.radius:
            self.radius = new_radius
            self.show_action_message(f"Brush size: {self.radius}")
    
    def toggle_theme(self):
        """Toggle between light and dark themes."""
        self.current_theme = 'dark' if self.current_theme == 'light' else 'light'
        self.colors = self.themes[self.current_theme]
        self.show_action_message(f"Switched to {self.current_theme} theme")
    
    def show_action_message(self, message, color=None):
        """Show a temporary action message."""
        self.action_message = message
        self.action_color = color if color else self.colors['text']
        self.last_action_time = time.time()
    
    def save_drawing(self):
        """Save the current drawing."""
        try:
            # Create directory if it doesn't exist
            import os
            if not os.path.exists('saved_drawings'):
                os.makedirs('saved_drawings')
            
            # Save with timestamp
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"saved_drawings/digit_{timestamp}.png"
            pygame.image.save(self.canvas, filename)
            self.show_action_message(f"Drawing saved as {filename}", self.colors['success'])
        except Exception as e:
            self.show_action_message(f"Error saving drawing: {e}", self.colors['error'])
    
    def predict(self):
        """Predict the drawn digit."""
        if self.model is None:
            self.show_action_message("Model not loaded!", self.colors['error'])
            return
        
        # Check if canvas is empty
        canvas_array = pygame.surfarray.array3d(self.canvas)
        if np.all(canvas_array == np.array(self.colors['canvas_bg'])):
            self.show_action_message("Canvas is empty! Draw something first.", self.colors['error'])
            return
            
        # Get the drawing as a numpy array
        img_array = pygame.surfarray.array3d(self.canvas)
        
        # Convert to grayscale and invert (MNIST has white digits on black background)
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img = cv2.flip(img, 1)
        
        # Preprocess for model input
        img_pil = Image.fromarray(img)
        img_pil = img_pil.resize((28, 28))
        self.processed_img = np.array(img_pil)
        
        # Normalize and reshape
        img_norm = self.processed_img.astype('float32') / 255.0
        img_input = img_norm.reshape(1, 28, 28, 1)
        
        # Make prediction
        prediction = self.model.predict(img_input, verbose=0)  # Suppress verbose output
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction) * 100
        
        # Store all confidences
        self.all_confidences = prediction[0] * 100
        
        # Update current prediction
        self.current_prediction = predicted_class
        self.current_confidence = confidence
        
        # Add to history
        self.prediction_history.insert(0, (predicted_class, confidence))
        if len(self.prediction_history) > self.max_history:
            self.prediction_history.pop()
        
        # Show action message
        confidence_color = self.colors['success'] if confidence > 70 else self.colors['error']
        self.show_action_message(
            f"Predicted: {predicted_class} (Confidence: {confidence:.1f}%)",
            confidence_color
        )
    
    def run(self):
        """Main application loop."""
        # Print instructions
        print("\nDigit Recognizer - Enhanced UI")
        print("================================")
        print("Instructions:")
        print("- Draw a digit (0-9) using the mouse")
        print("- Use the UI buttons or keyboard shortcuts:")
        print("  - C: Clear canvas")
        print("  - P: Predict digit")
        print("  - S: Save drawing")
        print("  - G: Toggle grid")
        print("  - T: Toggle between light/dark theme")
        print("  - +/-: Adjust brush size")
        print("\nNew Features:")
        print("- Modern UI with rounded buttons")
        print("- Light/dark theme support")
        print("- Optional grid for better drawing guidance")
        print("- Confidence visualization for all digits")
        print("- Improved canvas size and visual feedback")
        print("\nEnjoy drawing and recognizing digits!")
        
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                else:
                    self.handle_mouse_input(event)
                    self.handle_keyboard_input(event)
            
            self.draw()
        
        pygame.quit()

if __name__ == "__main__":
    app = DrawingApp()
    app.run()