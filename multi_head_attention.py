from manim import *
import numpy as np

class MultiHeadAttentionScene(Scene):
    def construct(self):
        # Configure some constants for the scene
        self.camera.background_color = "#1f303e"  # 3B1B-style dark blue background
        
        # ===== PART 1: INTRODUCTION =====
        self.introduce_mha()
        
        # ===== PART 2: TOKEN EMBEDDINGS =====
        token_mobjects, embedding_matrices = self.show_token_embeddings()
        
        # ===== PART 3: QUERY, KEY, VALUE TRANSFORMATIONS =====
        q_k_v_matrices = self.show_qkv_transformations(token_mobjects, embedding_matrices)
        
        # ===== PART 4: ATTENTION MECHANISM =====
        self.show_attention_mechanism(token_mobjects, q_k_v_matrices)
        
        # ===== PART 5: MULTI-LATENT ATTENTION =====
        self.show_mla_improvement()
        
        # ===== PART 6: CONCLUSION =====
        self.summarize()
        
    def introduce_mha(self):
        """Introduction to Multi-Head Attention"""
        # Clear screen to ensure no overlays from previous content
        self.clear()
        
        # Title slide
        title = Text("Multi-Head Attention", font_size=72)
        subtitle = Text("Understanding Q, K, V dimensions", font_size=48)
        subtitle.next_to(title, DOWN, buff=0.5)
        
        self.play(Write(title), run_time=1.5)
        self.wait(0.5)
        self.play(FadeIn(subtitle), run_time=1)
        self.wait(1.5)
        
        # Clear for next slide
        self.play(FadeOut(title), FadeOut(subtitle), run_time=1)
        self.wait(0.3)
        
        # New slide with smaller title
        slide_title = Text("Multi-Head Attention", font_size=48)
        slide_title.to_edge(UP, buff=0.5)
        
        self.play(Write(slide_title), run_time=1)
        
        # Explanation text
        explanation = Text(
            "In Multi-Head Attention, we have three key components:",
            font_size=36
        )
        explanation.next_to(slide_title, DOWN, buff=0.75)
        
        self.play(Write(explanation), run_time=1)
        self.wait(0.5)
        
        # Create the QKV elements - positioned with more space
        q_text = MathTex("\\text{Query (Q)}", font_size=44)
        k_text = MathTex("\\text{Key (K)}", font_size=44)
        v_text = MathTex("\\text{Value (V)}", font_size=44)
        
        # Position these elements with better spacing
        q_text.move_to(LEFT * 3.5 + UP * 1)
        k_text.move_to(ORIGIN + UP * 0)
        v_text.move_to(RIGHT * 3.5 + UP * 1)
        
        # Create explanation notes with shorter text to fit better
        q_note = Text("Represents current token", font_size=24, color=YELLOW)
        k_note = Text("Represents context tokens", font_size=24, color=GREEN)
        v_note = Text("Information to retrieve", font_size=24, color=BLUE)
        
        q_note.next_to(q_text, DOWN, buff=0.3)
        k_note.next_to(k_text, DOWN, buff=0.3)
        v_note.next_to(v_text, DOWN, buff=0.3)
        
        # Animation sequence - one at a time with pauses
        self.play(Write(q_text), run_time=0.8)
        self.play(FadeIn(q_note), run_time=0.5)
        self.wait(0.7)
        
        self.play(Write(k_text), run_time=0.8)
        self.play(FadeIn(k_note), run_time=0.5)
        self.wait(0.7)
        
        self.play(Write(v_text), run_time=0.8)
        self.play(FadeIn(v_note), run_time=0.5)
        self.wait(1.5)
        
        # Clear screen for next section
        self.play(
            FadeOut(explanation),
            FadeOut(q_text), FadeOut(q_note),
            FadeOut(k_text), FadeOut(k_note),
            FadeOut(v_text), FadeOut(v_note),
            run_time=1
        )
        
        # New slide for dimensions explanation
        dims_title = Text("Key Dimensions in Attention", font_size=40)
        dims_title.next_to(slide_title, DOWN, buff=0.75)
        
        self.play(Write(dims_title), run_time=1)
        self.wait(0.5)
        
        # Create dimension explanations as separate items for better spacing
        n_dim = MathTex("n", "=\\text{ number of tokens (sequence length)}", font_size=36)
        d_k_dim = MathTex("d_k", "=\\text{ dimension of keys}", font_size=36)
        d_v_dim = MathTex("d_v", "=\\text{ dimension of values}", font_size=36)
        
        n_dim.next_to(dims_title, DOWN, buff=0.5).align_to(dims_title, LEFT).shift(RIGHT * 0.5)
        d_k_dim.next_to(n_dim, DOWN, buff=0.4).align_to(n_dim, LEFT)
        d_v_dim.next_to(d_k_dim, DOWN, buff=0.4).align_to(d_k_dim, LEFT)
        
        # Show each dimension with a pause
        self.play(Write(n_dim), run_time=1)
        self.wait(0.7)
        self.play(Write(d_k_dim), run_time=1)
        self.wait(0.7)
        self.play(Write(d_v_dim), run_time=1)
        self.wait(1.5)
        
        # Clear screen for next section
        self.play(
            FadeOut(slide_title),
            FadeOut(dims_title),
            FadeOut(n_dim),
            FadeOut(d_k_dim),
            FadeOut(d_v_dim),
            run_time=1
        )
    
    def show_token_embeddings(self):
        """Show the tokens and their embeddings"""
        # Clear screen to ensure no overlays
        self.clear()
        
        # Title for this section
        section_title = Text("Token Embeddings", font_size=48)
        section_title.to_edge(UP, buff=0.5)
        
        self.play(Write(section_title), run_time=1)
        self.wait(0.5)
        
        # Create the sentence
        sentence_text = Text("\"I love deep learning\"", font_size=42)
        sentence_text.next_to(section_title, DOWN, buff=0.7)
        
        self.play(Write(sentence_text), run_time=1.5)
        self.wait(1)
        
        # Create token boxes with better spacing
        tokens = ["I", "love", "deep", "learning"]
        token_mobjects = []
        token_indices = []
        
        # Calculate better horizontal spacing
        total_width = 10  # Total width to distribute tokens
        token_spacing = total_width / (len(tokens) - 1) if len(tokens) > 1 else 0
        start_x = -total_width / 2 if len(tokens) > 1 else 0
        
        for i, token in enumerate(tokens):
            token_box = VGroup(
                Square(side_length=1, color=BLUE).set_fill(BLUE_E, opacity=0.3),
                Text(token, font_size=32)
            )
            
            # Position the tokens in a row with calculated spacing
            x_pos = start_x + (i * token_spacing)
            token_box.move_to([x_pos, 0.5, 0])
            
            # Create index labels
            index_label = Text(f"Token {i+1}", font_size=20, color=YELLOW)
            index_label.next_to(token_box, DOWN, buff=0.2)
            
            token_mobjects.append(token_box)
            token_indices.append(index_label)
        
        # Animate showing the tokens one by one
        for token_box, index in zip(token_mobjects, token_indices):
            self.play(FadeIn(token_box), FadeIn(index), run_time=0.75)
        
        # Indicate that n=4 (number of tokens)
        n_notation = MathTex("n = 4 \\text{ tokens}", font_size=36, color=YELLOW)
        n_notation.to_edge(LEFT).shift(UP * 0.5 + RIGHT * 1)
        
        self.play(Write(n_notation), run_time=1)
        self.wait(1.5)
        
        # Clear for next slide while keeping the tokens
        self.play(
            FadeOut(sentence_text),
            *[FadeOut(index) for index in token_indices],
            run_time=0.8
        )
        
        # New title for embeddings
        embed_title = Text("Converting Tokens to Embeddings", font_size=36)
        embed_title.next_to(section_title, DOWN, buff=0.7)
        
        self.play(Write(embed_title), run_time=1)
        
        # Explanation of embeddings
        embed_explanation = Text("Each token is converted to a numerical vector", font_size=28)
        embed_explanation.next_to(embed_title, DOWN, buff=0.4)
        
        self.play(Write(embed_explanation), run_time=1)
        self.wait(0.7)
        
        # Show embedding dimension
        d_model_notation = MathTex("d_{model} = 8", font_size=36, color=GREEN)
        d_model_notation.next_to(n_notation, RIGHT, buff=3)
        
        self.play(Write(d_model_notation), run_time=1)
        self.wait(1)
        
        # Create sample embedding matrices (4x8 for simplicity)
        embeddings = [
            np.array([0.1, 0.3, 0.2, 0.0, 0.4, 0.1, 0.3, 0.2]),  # I
            np.array([0.0, 0.2, 0.3, 0.4, 0.5, 0.2, 0.1, 0.0]),  # love
            np.array([0.4, 0.0, 0.1, 0.2, 0.3, 0.2, 0.4, 0.5]),  # deep
            np.array([0.3, 0.1, 0.4, 0.5, 0.0, 0.3, 0.2, 0.1])   # learning
        ]
        
        embedding_matrices = []
        
        # Show embeddings one by one with better positioning
        for i, (token_box, embed) in enumerate(zip(token_mobjects, embeddings)):
            # Create a matrix visualization for each embedding
            matrix = Matrix(
                [[f"{val:.1f}"] for val in embed],
                v_buff=0.6,
                h_buff=0.8,
                bracket_h_buff=0.1,
                bracket_v_buff=0.1,
                element_to_mobject_config={"font_size": 20}
            ).scale(0.6)
            
            # Position the matrix below its token with enough space
            matrix.next_to(token_box, DOWN, buff=0.7)
            embedding_matrices.append(matrix)
            
            # Animate the embedding matrix formation
            self.play(
                token_box.animate.set_color(YELLOW),
                TransformFromCopy(token_box, matrix),
                run_time=1
            )
            self.play(token_box.animate.set_color(BLUE), run_time=0.3)
            self.wait(0.5)
        
        self.wait(1.5)
        
        # Clear screen for next section
        self.play(
            FadeOut(section_title),
            FadeOut(embed_title),
            FadeOut(embed_explanation),
            FadeOut(n_notation),
            FadeOut(d_model_notation),
            *[FadeOut(token) for token in token_mobjects],
            *[FadeOut(matrix) for matrix in embedding_matrices],
            run_time=1
        )
        
        # Keep the elements on screen but move them up to make room for the next section
        grouped_elements = VGroup(*token_mobjects, *embedding_matrices)
        
        self.play(
            grouped_elements.animate.scale(0.8).to_edge(UP, buff=1),
            FadeOut(sentence_text),
            run_time=1
        )
        
        return token_mobjects, embedding_matrices
    
    def show_qkv_transformations(self, token_mobjects, embedding_matrices):
        """Show transformation of embeddings into Q, K, V vectors"""
        # Title for this section
        section_title = Text("Transforming Embeddings into Q, K, V", font_size=40)
        section_title.to_edge(UP, buff=0.2)
        
        self.play(Write(section_title), run_time=1)
        
        # Focus on the "love" token (index 1) for demonstration
        focus_index = 1  # "love" token
        
        # Highlight the token we're focusing on
        self.play(
            token_mobjects[focus_index].animate.set_color(YELLOW).scale(1.1),
            run_time=0.7
        )
        
        # Define the transformation process
        transform_eq = MathTex(
            "\\text{For token }",
            "\\text{``love''}:",
            font_size=36
        )
        transform_eq.next_to(embedding_matrices[focus_index], DOWN, buff=1)
        
        self.play(Write(transform_eq), run_time=1)
        
        # Show the transformation equations
        q_eq = MathTex("Q_{love} = W_Q \\times \\text{embedding}_{love}", font_size=32)
        k_eq = MathTex("K_{love} = W_K \\times \\text{embedding}_{love}", font_size=32)
        v_eq = MathTex("V_{love} = W_V \\times \\text{embedding}_{love}", font_size=32)
        
        q_eq.next_to(transform_eq, DOWN, buff=0.5).shift(LEFT * 3)
        k_eq.next_to(q_eq, DOWN, buff=0.3)
        v_eq.next_to(k_eq, DOWN, buff=0.3)
        
        self.play(Write(q_eq), run_time=0.8)
        self.play(Write(k_eq), run_time=0.8)
        self.play(Write(v_eq), run_time=0.8)
        self.wait(1)
        
        # Show the resulting Q, K, V vectors (simplified to 4 dimensions)
        # Note: In practice, these would be calculated from linear transformations
        q_vec = [0.1, 0.2, 0.0, 0.1]
        k_vec = [0.3, 0.1, 0.0, 0.2]
        v_vec = [0.4, 0.1, 0.3, 0.2]
        
        # Create vector visualizations
        q_matrix = Matrix(
            [[f"{val:.1f}"] for val in q_vec],
            v_buff=0.6,
            h_buff=0.8,
            element_to_mobject_config={"font_size": 24}
        ).scale(0.5)
        
        k_matrix = Matrix(
            [[f"{val:.1f}"] for val in k_vec],
            v_buff=0.6,
            h_buff=0.8,
            element_to_mobject_config={"font_size": 24}
        ).scale(0.5)
        
        v_matrix = Matrix(
            [[f"{val:.1f}"] for val in v_vec],
            v_buff=0.6,
            h_buff=0.8,
            element_to_mobject_config={"font_size": 24}
        ).scale(0.5)
        
        # Position the vectors
        q_matrix.next_to(q_eq, RIGHT, buff=1)
        k_matrix.next_to(k_eq, RIGHT, buff=1)
        v_matrix.next_to(v_eq, RIGHT, buff=1)
        
        # Label the dimensions
        d_k_label = MathTex("d_k = 4", font_size=28, color=GREEN)
        d_v_label = MathTex("d_v = 4", font_size=28, color=BLUE)
        
        d_k_label.next_to(q_matrix, RIGHT, buff=0.5)
        d_v_label.next_to(v_matrix, RIGHT, buff=0.5)
        
        # Animate the vectors appearing
        self.play(Create(q_matrix), run_time=0.7)
        self.play(Create(k_matrix), run_time=0.7)
        self.play(Create(v_matrix), run_time=0.7)
        
        self.play(Write(d_k_label), Write(d_v_label), run_time=0.7)
        self.wait(1)
        
        # Explanation that this happens for all tokens
        all_tokens_note = Text(
            "This happens for all tokens, creating Q, K, V for each",
            font_size=28
        )
        all_tokens_note.next_to(v_eq, DOWN, buff=0.8)
        
        self.play(Write(all_tokens_note), run_time=1)
        self.wait(1)
        
        # Create complete K and V matrices for all tokens
        k_all_label = MathTex("K_{all} = \\begin{bmatrix} K_I \\\\ K_{love} \\\\ K_{deep} \\\\ K_{learning} \\end{bmatrix}", font_size=32)
        v_all_label = MathTex("V_{all} = \\begin{bmatrix} V_I \\\\ V_{love} \\\\ V_{deep} \\\\ V_{learning} \\end{bmatrix}", font_size=32)
        
        k_all_label.shift(RIGHT * 3 + UP * 0.5)
        v_all_label.next_to(k_all_label, DOWN, buff=0.5)
        
        k_dims = MathTex("\\text{shape: } (4 \\times 4)", font_size=28, color=GREEN)
        v_dims = MathTex("\\text{shape: } (4 \\times 4)", font_size=28, color=BLUE)
        
        k_dims.next_to(k_all_label, RIGHT, buff=0.5)
        v_dims.next_to(v_all_label, RIGHT, buff=0.5)
        
        self.play(
            Write(k_all_label),
            Write(v_all_label),
            run_time=1.2
        )
        
        self.play(
            Write(k_dims),
            Write(v_dims),
            run_time=0.8
        )
        self.wait(1)
        
        # Store matrices for the next section
        q_k_v_matrices = {
            "q_matrix": q_matrix,
            "k_matrix": k_matrix,
            "v_matrix": v_matrix,
            "k_all": k_all_label,
            "v_all": v_all_label,
            "q_eq": q_eq,
            "k_eq": k_eq,
            "v_eq": v_eq
        }
        
        # Clear screen for next section
        self.play(
            FadeOut(section_title),
            FadeOut(transform_eq),
            FadeOut(q_eq), FadeOut(q_matrix),
            FadeOut(k_eq), FadeOut(k_matrix),
            FadeOut(v_eq), FadeOut(v_matrix),
            FadeOut(d_k_label), FadeOut(d_v_label),
            FadeOut(all_tokens_note),
            FadeOut(k_all_label), FadeOut(k_dims),
            FadeOut(v_all_label), FadeOut(v_dims),
            *[FadeOut(mob) for mob in token_mobjects],
            *[FadeOut(mat) for mat in embedding_matrices],
            run_time=1
        )
        
        return q_k_v_matrices
    
    def show_attention_mechanism(self, token_mobjects, q_k_v_matrices):
        """Show how attention works for the next token generation"""
        # Title for this section
        section_title = Text("Computing Attention for Next Token", font_size=40)
        section_title.to_edge(UP, buff=0.2)
        
        self.play(Write(section_title), run_time=1)
        
        # Set up the token sequence again
        tokens = ["I", "love", "deep", "learning", "?"]
        token_row = VGroup()
        
        for i, token_text in enumerate(tokens):
            token_box = VGroup(
                Square(side_length=1, color=BLUE if i < 4 else YELLOW).set_fill(
                    BLUE_E if i < 4 else YELLOW_E, opacity=0.3
                ),
                Text(token_text, font_size=32)
            )
            token_box.shift(LEFT * 5 + RIGHT * (i * 2.5))
            token_row.add(token_box)
        
        token_row.center().shift(UP * 1.5)
        
        self.play(FadeIn(token_row), run_time=1)
        
        # Highlight the question mark as the next token to generate
        next_token_label = Text("Next token to generate", font_size=28, color=YELLOW)
        next_token_label.next_to(token_row[4], DOWN, buff=0.3)
        
        self.play(
            token_row[4].animate.set_color(YELLOW).scale(1.1),
            Write(next_token_label),
            run_time=0.8
        )
        self.wait(0.5)
        
        # Create Query vector for the next token
        q_next_eq = MathTex("Q_{next} = [0.2, 0.1, 0.0, 0.3]", font_size=32)
        q_next_eq.next_to(next_token_label, DOWN, buff=0.5)
        
        self.play(Write(q_next_eq), run_time=1)
        self.wait(0.5)
        
        # Show attention score computation
        attention_title = Text("Computing Attention Scores:", font_size=30)
        attention_title.next_to(q_next_eq, DOWN, buff=0.8)
        
        self.play(Write(attention_title), run_time=0.8)
        
        # Attention formula
        attention_formula = MathTex(
            "\\text{score} = \\frac{Q_{next} \\cdot K^\\top_{past}}{\\sqrt{d_k}}",
            font_size=36
        )
        attention_formula.next_to(attention_title, DOWN, buff=0.4)
        
        self.play(Write(attention_formula), run_time=1)
        self.wait(0.7)
        
        # Show individual dot products with each past key
        dot_products = [
            MathTex("Q_{next} \\cdot K_I = 0.20", font_size=30),
            MathTex("Q_{next} \\cdot K_{love} = 0.30", font_size=30),
            MathTex("Q_{next} \\cdot K_{deep} = 0.60 \\text{ (highest)}", font_size=30, color=YELLOW),
            MathTex("Q_{next} \\cdot K_{learning} = 0.10", font_size=30)
        ]
        
        for i, dot_prod in enumerate(dot_products):
            dot_prod.next_to(attention_formula, DOWN, buff=0.6).shift(DOWN * i * 0.5)
            self.play(Write(dot_prod), run_time=0.6)
            
            # Highlight the token with a brief pulse
            self.play(
                token_row[i].animate.set_color(RED),
                run_time=0.3
            )
            self.play(
                token_row[i].animate.set_color(BLUE),
                run_time=0.3
            )
        
        self.wait(0.5)
        
        # Show softmax operation
        softmax_title = Text("Applying Softmax:", font_size=30)
        softmax_title.next_to(dot_products[-1], DOWN, buff=0.7)
        
        softmax_eq = MathTex(
            "\\text{attention\\_weights} = \\text{softmax}([0.20, 0.30, 0.60, 0.10])",
            font_size=30
        )
        softmax_eq.next_to(softmax_title, DOWN, buff=0.3)
        
        softmax_result = MathTex(
            "= [0.21, 0.24, 0.43, 0.12]",
            font_size=30
        )
        softmax_result.next_to(softmax_eq, DOWN, buff=0.3)
        
        self.play(Write(softmax_title), run_time=0.7)
        self.play(Write(softmax_eq), run_time=0.8)
        self.play(Write(softmax_result), run_time=0.7)
        self.wait(1)
        
        # Prepare for the next part by clearing some elements
        self.play(
            *[FadeOut(dot_prod) for dot_prod in dot_products],
            FadeOut(attention_title),
            FadeOut(attention_formula),
            run_time=0.7
        )
        
        # Compute the context vector using value vectors
        context_title = Text("Computing the Context Vector:", font_size=30)
        context_title.next_to(softmax_result, DOWN, buff=0.7)
        
        context_eq = MathTex(
            "\\text{context} = 0.21 \\times V_I + 0.24 \\times V_{love} + 0.43 \\times V_{deep} + 0.12 \\times V_{learning}",
            font_size=28
        )
        context_eq.next_to(context_title, DOWN, buff=0.3)
        
        self.play(Write(context_title), run_time=0.7)
        self.play(Write(context_eq), run_time=1.2)
        self.wait(1)
        
        # Visualize the weighted sum with highlighted weights
        weighted_values = [
            (0.21, "I", BLUE_C),
            (0.24, "love", BLUE_C),
            (0.43, "deep", YELLOW),  # Highest weight
            (0.12, "learning", BLUE_C)
        ]
        
        context_visual = VGroup()
        
        for i, (weight, token, color) in enumerate(weighted_values):
            # Create a weighted representation
            weight_text = Text(f"{weight:.2f} ×", font_size=28, color=color)
            token_text = Text(token, font_size=28)
            
            combo = VGroup(weight_text, token_text)
            combo.arrange(RIGHT, buff=0.2)
            
            # Position in a vertical stack
            if i == 0:
                combo.next_to(context_eq, DOWN, buff=0.7)
            else:
                combo.next_to(context_visual[-1], DOWN, buff=0.3)
            
            context_visual.add(combo)
            
            self.play(FadeIn(combo), run_time=0.5)
            
            # Highlight the corresponding token in the row
            self.play(
                token_row[i].animate.set_color(color).scale(1.1 if color == YELLOW else 1.0),
                run_time=0.3
            )
            self.play(
                token_row[i].animate.set_color(BLUE).scale(1/1.1 if color == YELLOW else 1.0),
                run_time=0.3
            )
        
        # Show final result - predicting the next token
        next_token_prediction = Text("⟹ Predicts next token: \"models\"", font_size=32, color=GREEN)
        next_token_prediction.next_to(context_visual, DOWN, buff=0.7)
        
        self.play(Write(next_token_prediction), run_time=1)
        self.wait(1.5)
        
        # Clear screen for next section
        self.play(
            FadeOut(section_title),
            FadeOut(token_row),
            FadeOut(next_token_label),
            FadeOut(q_next_eq),
            FadeOut(softmax_title),
            FadeOut(softmax_eq),
            FadeOut(softmax_result),
            FadeOut(context_title),
            FadeOut(context_eq),
            *[FadeOut(item) for item in context_visual],
            FadeOut(next_token_prediction),
            run_time=1
        )
    
    def show_mla_improvement(self):
        """Show how MLA improves efficiency by compressing K, V matrices"""
        # Clear screen to ensure no overlays
        self.clear()
        
        # Title for this section
        section_title = Text("Multi-Latent Attention (MLA)", font_size=48)
        section_title.to_edge(UP, buff=0.5)
        
        self.play(Write(section_title), run_time=1)
        self.wait(0.5)
        
        # Slide 1: Explain the problem with standard attention
        problem_title = Text("The Memory Problem", font_size=36, color=RED)
        problem_title.next_to(section_title, DOWN, buff=0.7)
        
        self.play(Write(problem_title), run_time=0.8)
        self.wait(0.3)
        
        problem_text = Text(
            "Standard attention requires storing all past Key and Value vectors",
            font_size=28
        )
        problem_text.next_to(problem_title, DOWN, buff=0.5)
        
        self.play(Write(problem_text), run_time=1)
        self.wait(0.7)
        
        # Visualize large K, V matrices for long sequences
        large_seq_text = Text("For long sequences (e.g., 32,000 tokens):", font_size=26)
        large_seq_text.next_to(problem_text, DOWN, buff=0.6)
        
        self.play(Write(large_seq_text), run_time=0.8)
        self.wait(0.5)
        
        # Create matrix visualizations - with better positioning
        k_matrix_large = Rectangle(
            height=3, width=1.2, 
            color=BLUE
        ).shift(LEFT * 3 + DOWN * 1)
        
        v_matrix_large = Rectangle(
            height=3, width=1.2,
            color=GREEN
        ).shift(RIGHT * 3 + DOWN * 1)
        
        k_label = MathTex("K_{past}", font_size=36)
        v_label = MathTex("V_{past}", font_size=36)
        
        k_label.next_to(k_matrix_large, UP, buff=0.3)
        v_label.next_to(v_matrix_large, UP, buff=0.3)
        
        k_dim = MathTex("(32,000 \\times 128)", font_size=24, color=YELLOW)
        v_dim = MathTex("(32,000 \\times 128)", font_size=24, color=YELLOW)
        
        k_dim.next_to(k_matrix_large, DOWN, buff=0.3)
        v_dim.next_to(v_matrix_large, DOWN, buff=0.3)
        
        self.play(
            Create(k_matrix_large),
            Create(v_matrix_large),
            run_time=1
        )
        
        self.play(
            Write(k_label),
            Write(v_label),
            run_time=0.8
        )
        
        self.play(
            Write(k_dim),
            Write(v_dim),
            run_time=0.8
        )
        
        # Add memory consumption warning
        memory_text = Text("Huge memory consumption!", font_size=28, color=RED)
        memory_text.next_to(v_matrix_large, DOWN, buff=1.0).shift(LEFT * 3)
        
        self.play(Write(memory_text), run_time=0.7)
        self.wait(1.5)
        
        # Clear for next slide but keep title
        self.play(
            FadeOut(problem_title),
            FadeOut(problem_text),
            FadeOut(large_seq_text),
            FadeOut(k_matrix_large), 
            FadeOut(k_label), 
            FadeOut(k_dim),
            FadeOut(v_matrix_large), 
            FadeOut(v_label), 
            FadeOut(v_dim),
            FadeOut(memory_text),
            run_time=1
        )
        
        # Slide 2: Show MLA solution
        solution_title = Text("The MLA Solution", font_size=36, color=GREEN)
        solution_title.next_to(section_title, DOWN, buff=0.7)
        
        self.play(Write(solution_title), run_time=0.8)
        self.wait(0.3)
        
        solution_text = Text(
            "Compress Keys and Values into smaller latent vectors",
            font_size=28
        )
        solution_text.next_to(solution_title, DOWN, buff=0.5)
        
        self.play(Write(solution_text), run_time=1)
        self.wait(0.7)
        
        # Create before and after comparison
        before_title = Text("Before:", font_size=24, color=RED)
        after_title = Text("After:", font_size=24, color=GREEN)
        
        before_title.to_edge(LEFT, buff=2).shift(UP * 0.5)
        after_title.to_edge(RIGHT, buff=2).shift(UP * 0.5)
        
        self.play(Write(before_title), Write(after_title), run_time=0.7)
        
        # Before matrices (smaller than previous slide for better comparison)
        k_matrix_before = Rectangle(height=2, width=1, color=BLUE).next_to(before_title, DOWN, buff=0.4).shift(LEFT * 1)
        v_matrix_before = Rectangle(height=2, width=1, color=GREEN).next_to(before_title, DOWN, buff=0.4).shift(RIGHT * 1)
        
        k_before_dim = MathTex("(32,000 \\times 128)", font_size=20, color=YELLOW)
        v_before_dim = MathTex("(32,000 \\times 128)", font_size=20, color=YELLOW)
        
        k_before_dim.next_to(k_matrix_before, DOWN, buff=0.2)
        v_before_dim.next_to(v_matrix_before, DOWN, buff=0.2)
        
        # After matrices
        k_matrix_after = Rectangle(height=0.8, width=1, color=BLUE).next_to(after_title, DOWN, buff=0.4).shift(LEFT * 1)
        v_matrix_after = Rectangle(height=0.8, width=1, color=GREEN).next_to(after_title, DOWN, buff=0.4).shift(RIGHT * 1)
        
        k_after_dim = MathTex("(512 \\times 128)", font_size=20, color=YELLOW)
        v_after_dim = MathTex("(512 \\times 128)", font_size=20, color=YELLOW)
        
        k_after_dim.next_to(k_matrix_after, DOWN, buff=0.2)
        v_after_dim.next_to(v_matrix_after, DOWN, buff=0.2)
        
        # Show before matrices
        self.play(
            Create(k_matrix_before),
            Create(v_matrix_before),
            run_time=0.8
        )
        
        self.play(
            Write(k_before_dim),
            Write(v_before_dim),
            run_time=0.7
        )
        
        # Show after matrices
        self.play(
            Create(k_matrix_after),
            Create(v_matrix_after),
            run_time=0.8
        )
        
        self.play(
            Write(k_after_dim),
            Write(v_after_dim),
            run_time=0.7
        )
        
        # Add compression arrows
        k_arrow = Arrow(
            k_matrix_before.get_right(), 
            k_matrix_after.get_left(),
            buff=0.2, color=YELLOW
        )
        
        v_arrow = Arrow(
            v_matrix_before.get_right(), 
            v_matrix_after.get_left(),
            buff=0.2, color=YELLOW
        )
        
        self.play(
            Create(k_arrow),
            Create(v_arrow),
            run_time=0.8
        )
        
        self.wait(1)
        
        # Slide 3: Explain the benefits
        benefits_group = VGroup()
        
        # Clear previous elements but keep the title
        self.play(
            FadeOut(solution_title),
            FadeOut(solution_text),
            FadeOut(before_title),
            FadeOut(after_title),
            FadeOut(k_matrix_before),
            FadeOut(v_matrix_before),
            FadeOut(k_before_dim),
            FadeOut(v_before_dim),
            FadeOut(k_matrix_after),
            FadeOut(v_matrix_after),
            FadeOut(k_after_dim),
            FadeOut(v_after_dim),
            FadeOut(k_arrow),
            FadeOut(v_arrow),
            run_time=1
        )
        
        benefits_title = Text("Benefits of MLA", font_size=36, color=GREEN)
        benefits_title.next_to(section_title, DOWN, buff=0.7)
        
        self.play(Write(benefits_title), run_time=0.8)
        self.wait(0.5)
        
        # List benefits one by one
        benefits = [
            "Queries attend to 512 latent vectors instead of 32,000 tokens",
            "Memory usage reduced by ~98.4%",
            "Faster inference speed",
            "Enables much longer context windows"
        ]
        
        benefit_mobjects = []
        for i, benefit in enumerate(benefits):
            bullet = Text("•", font_size=28, color=YELLOW)
            text = Text(benefit, font_size=28)
            
            row = VGroup(bullet, text)
            row.arrange(RIGHT, buff=0.3, aligned_edge=UP)
            
            if i == 0:
                row.next_to(benefits_title, DOWN, buff=0.8, aligned_edge=LEFT)
                row.shift(RIGHT * 0.5)  # Indent slightly
            else:
                row.next_to(benefit_mobjects[-1], DOWN, buff=0.5, aligned_edge=LEFT)
            
            benefit_mobjects.append(row)
            
            self.play(Write(bullet), Write(text), run_time=0.8)
            self.wait(0.5)
        
        self.wait(1.5)
        
        # Clear screen for conclusion
        self.play(
            FadeOut(section_title),
            FadeOut(benefits_title),
            *[FadeOut(item) for item in benefit_mobjects],
            run_time=1
        )
    
    def summarize(self):
        """Summary of key concepts of MHA and MLA"""
        # Title
        title = Text("Multi-Head Attention: Key Takeaways", font_size=48)
        title.to_edge(UP, buff=0.5)
        
        self.play(Write(title), run_time=1)
        self.wait(0.5)
        
        # Create key takeaways
        takeaways = [
            "Q, K, V are the fundamental components of attention",
            "n = number of tokens (context length)",
            "d_k, d_v = dimensions of keys and values",
            "Q matches with K to find relevant context",
            "Context vector is weighted sum of V vectors",
            "MLA compresses Keys and Values to save memory"
        ]
        
        takeaway_texts = VGroup()
        
        for i, point in enumerate(takeaways):
            bullet = Text("•", font_size=32, color=YELLOW)
            text = Text(point, font_size=32)
            
            row = VGroup(bullet, text)
            row.arrange(RIGHT, buff=0.3, aligned_edge=UP)
            
            if i == 0:
                row.next_to(title, DOWN, buff=0.8, aligned_edge=LEFT)
                row.shift(RIGHT * 0.5)  # Indent slightly
            else:
                row.next_to(takeaway_texts[-1], DOWN, buff=0.3, aligned_edge=LEFT)
            
            takeaway_texts.add(row)
            
            self.play(Write(bullet), Write(text), run_time=0.8)
            self.wait(0.3)
        
        self.wait(1)
        
        # Final thoughts
        final_note = Text(
            "Multi-Latent Attention enables efficient inference\nwith long context windows",
            font_size=36, color=GREEN
        )
        final_note.next_to(takeaway_texts, DOWN, buff=1)
        
        self.play(Write(final_note), run_time=1.5)
        self.wait(2)
        
        # Fade out everything
        self.play(
            FadeOut(title),
            *[FadeOut(item) for item in takeaway_texts],
            FadeOut(final_note),
            run_time=1.5
        )
        
        # Final title
        thanks = Text("Thanks for watching!", font_size=72, color=YELLOW)
        
        self.play(Write(thanks), run_time=1.5)
        self.wait(2)
        self.play(FadeOut(thanks), run_time=1)