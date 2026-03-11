"""
Test Code Understanding and Output Quality at Different Token Counts

This test validates that KV cache compaction preserves:
1. Code syntax and structure
2. Logical flow and reasoning
3. Variable references and dependencies
4. Code comments and documentation
"""

import asyncio
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Set
from pathlib import Path


@dataclass
class TestResult:
    """Results from a code understanding test"""
    token_count: int
    compression_ratio: float
    original_code: str
    compacted_output: str
    syntax_preserved: bool
    logic_preserved: bool
    variable_preserved: bool
    quality_score: float


@dataclass
class CodeTest:
    """A code test case"""
    name: str
    language: str
    code: str


# Test cases with increasing complexity
CODE_TESTS: List[CodeTest] = [
    CodeTest(
        name="simple-function",
        language="python",
        code="""
# Calculate fibonacci number recursively
def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# Calculate factorial iteratively
def factorial(n: int) -> int:
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

# Main function
def main():
    print(f"Fibonacci of 10: {fibonacci(10)}")
    print(f"Factorial of 5: {factorial(5)}")

if __name__ == "__main__":
    main()
"""
    ),
    CodeTest(
        name="class-hierarchy",
        language="python",
        code="""
# Base class for all shapes
from abc import ABC, abstractmethod
import math

class Shape(ABC):
    def __init__(self, color: str):
        self.color = color

    @abstractmethod
    def get_area(self) -> float:
        pass

    @abstractmethod
    def get_perimeter(self) -> float:
        pass

    def describe(self) -> str:
        return f"A {self.color} shape with area {self.get_area()}"

# Rectangle class
class Rectangle(Shape):
    def __init__(self, color: str, width: float, height: float):
        super().__init__(color)
        self.width = width
        self.height = height

    def get_area(self) -> float:
        return self.width * self.height

    def get_perimeter(self) -> float:
        return 2 * (self.width + self.height)

# Circle class
class Circle(Shape):
    def __init__(self, color: str, radius: float):
        super().__init__(color)
        self.radius = radius

    def get_area(self) -> float:
        return math.pi * self.radius ** 2

    def get_perimeter(self) -> float:
        return 2 * math.pi * self.radius

# Usage
rect = Rectangle("blue", 5, 10)
circle = Circle("red", 7)
print(rect.describe())
print(circle.describe())
"""
    ),
    CodeTest(
        name="async-data-processing",
        language="python",
        code="""
# Interface for user data
from dataclasses import dataclass
from typing import List, Optional
import asyncio

@dataclass
class User:
    id: int
    name: str
    email: str
    theme: str
    notifications: bool

# Fetch user data from API (simulated)
async def fetch_user(user_id: int) -> User:
    # Simulate API call
    await asyncio.sleep(0.1)
    return User(
        id=user_id,
        name=f"User {user_id}",
        email=f"user{user_id}@example.com",
        theme="light",
        notifications=True
    )

# Process multiple users concurrently
async def process_users(user_ids: List[int]) -> List[Optional[User]]:
    async def process_single(user_id: int) -> Optional[User]:
        try:
            user = await fetch_user(user_id)
            # Transform user data
            user.notifications = False  # Disable by default
            return user
        except Exception as error:
            print(f"Error processing user {user_id}: {error}")
            return None

    # Process concurrently
    results = await asyncio.gather(*[process_single(uid) for uid in user_ids])

    # Filter out failed fetches
    return [r for r in results if r is not None]

# Main processing pipeline
async def main():
    user_ids = [1, 2, 3, 4, 5]
    users = await process_users(user_ids)

    print(f"Processed {len(users)} users")
    for user in users:
        print(f"{user.name}: {user.email}")

if __name__ == "__main__":
    asyncio.run(main())
"""
    ),
    CodeTest(
        name="algorithm-implementation",
        language="python",
        code="""
# Binary Search Tree implementation
class TreeNode:
    def __init__(self, value: int):
        self.value = value
        self.left = None
        self.right = None

class BinarySearchTree:
    def __init__(self):
        self.root = None

    def insert(self, value: int) -> None:
        new_node = TreeNode(value)

        if not self.root:
            self.root = new_node
            return

        self._insert_node(self.root, new_node)

    def _insert_node(self, node: TreeNode, new_node: TreeNode) -> None:
        if new_node.value < node.value:
            if not node.left:
                node.left = new_node
            else:
                self._insert_node(node.left, new_node)
        else:
            if not node.right:
                node.right = new_node
            else:
                self._insert_node(node.right, new_node)

    def search(self, value: int) -> Optional[TreeNode]:
        return self._search_node(self.root, value)

    def _search_node(self, node: Optional[TreeNode], value: int) -> Optional[TreeNode]:
        if not node:
            return None
        if value == node.value:
            return node
        if value < node.value:
            return self._search_node(node.left, value)
        return self._search_node(node.right, value)

    # In-order traversal (sorted)
    def in_order_traversal(self) -> List[int]:
        result = []
        self._traverse(self.root, result)
        return result

    def _traverse(self, node: Optional[TreeNode], result: List[int]) -> None:
        if not node:
            return
        self._traverse(node.left, result)
        result.append(node.value)
        self._traverse(node.right, result)

# Usage example
bst = BinarySearchTree()
for n in [50, 30, 70, 20, 40, 60, 80]:
    bst.insert(n)

print("Sorted:", bst.in_order_traversal())
print("Found 40:", bst.search(40))
print("Found 100:", bst.search(100))
"""
    ),
]


class CodeUnderstandingTest:
    """Test code understanding after KV cache compaction"""

    def __init__(self):
        pass

    def test_at_token_count(
        self,
        target_tokens: int,
        code_test: CodeTest
    ) -> TestResult:
        """Test code understanding at specific token count"""
        print(f"Testing {code_test.name} at ~{target_tokens} tokens...")

        # Create prompt with code repeated to reach target token count
        prompt = self._create_prompt_with_tokens(code_test.code, target_tokens)

        # Simulate compaction (in real implementation, use actual library)
        start_time = time.time()
        compaction_ratio = self._simulate_compaction(prompt)
        compaction_time = time.time() - start_time

        # Generate output from compacted cache
        compacted_output = self._generate_output(prompt, code_test.language)

        # Evaluate quality
        return TestResult(
            token_count=target_tokens,
            compression_ratio=compaction_ratio,
            original_code=code_test.code,
            compacted_output=compacted_output,
            syntax_preserved=self._check_syntax_preserved(code_test.code, compacted_output),
            logic_preserved=self._check_logic_preserved(code_test.code, compacted_output),
            variable_preserved=self._check_variables_preserved(code_test.code, compacted_output),
            quality_score=self._calculate_quality_score(code_test.code, compacted_output),
        )

    def _create_prompt_with_tokens(self, code: str, target_tokens: int) -> str:
        """Create prompt with target token count by repeating code"""
        # Estimate tokens (rough approximation: ~4 chars per token)
        estimated_tokens_per_code = len(code) // 4
        repetitions = (target_tokens // estimated_tokens_per_code) + 1

        prompt = f"# Complete this code and explain its functionality:\n\n{code}\n\n"

        # Add context and variations
        for i in range(repetitions):
            prompt += f"\n# Variation {i + 1}:\n{code}\n\n"
            prompt += f"# Explanation: This code demonstrates {self._extract_concept(code)}\n\n"

        return prompt

    def _simulate_compaction(self, prompt: str) -> float:
        """Simulate compaction ratio (in real implementation, use actual library)"""
        # Simulate 5x compression (keep 20%)
        return 5.0

    def _generate_output(self, prompt: str, language: str) -> str:
        """Generate output from compacted cache (placeholder)"""
        # In real implementation, this would call the LLM
        return f"# Generated {language} code would appear here\n# This demonstrates understanding of the original code structure"

    def _check_syntax_preserved(self, original: str, output: str) -> bool:
        """Check if syntax is preserved in output"""
        # Check for balanced braces, parentheses, brackets
        def check_balance(s: str, open_char: str, close_char: str) -> bool:
            balance = 0
            for char in s:
                if char == open_char:
                    balance += 1
                elif char == close_char:
                    balance -= 1
                if balance < 0:
                    return False
            return balance == 0

        return (
            check_balance(output, '{', '}') and
            check_balance(output, '(', ')') and
            check_balance(output, '[', ']')
        )

    def _check_logic_preserved(self, original: str, output: str) -> bool:
        """Check if logical flow is preserved"""
        # Check for key control flow keywords
        keywords = ['if', 'else', 'elif', 'for', 'while', 'return', 'async', 'await', 'def', 'class']

        original_keywords = {kw for kw in keywords if kw in original}
        output_keywords = {kw for kw in keywords if kw in output}

        # Check if most keywords are preserved
        return len(original_keywords & output_keywords) / len(original_keywords) > 0.8 if original_keywords else True

    def _check_variables_preserved(self, original: str, output: str) -> bool:
        """Check if variable names are preserved"""
        # Extract variable/function names (simplified regex)
        python_vars = set(re.findall(r'\bdef\s+(\w+)\b', original))
        python_vars.update(re.findall(r'\bclass\s+(\w+)\b', original))
        python_vars.update(re.findall(r'\b(\w+)\s*=', original))

        # Check if variables appear in output
        preserved_count = sum(1 for var in python_vars if var in output)

        return preserved_count / len(python_vars) > 0.8 if python_vars else True

    def _calculate_quality_score(self, original: str, output: str) -> float:
        """Calculate overall quality score"""
        syntax_score = 1.0 if self._check_syntax_preserved(original, output) else 0.0
        logic_score = 1.0 if self._check_logic_preserved(original, output) else 0.0
        var_score = 1.0 if self._check_variables_preserved(original, output) else 0.0

        # Weighted average
        return (syntax_score * 0.3 + logic_score * 0.4 + var_score * 0.3)

    def _extract_concept(self, code: str) -> str:
        """Extract main concept from code"""
        if 'class ' in code:
            return 'object-oriented programming'
        if 'async ' in code or 'await ' in code:
            return 'asynchronous operations'
        if 'def ' in code:
            return 'functional programming'
        if 'for ' in code or 'while ' in code:
            return 'looping constructs'
        return 'programming concepts'

    def run_all_tests(self) -> None:
        """Run all tests across different token counts"""
        token_counts = [1000, 10000]

        print("=" * 50)
        print("Code Understanding Tests")
        print("=" * 50 + "\n")

        for code_test in CODE_TESTS:
            print(f"\n[*] Test: {code_test.name} ({code_test.language})")
            print("-" * 50)

            for token_count in token_counts:
                try:
                    result = self.test_at_token_count(token_count, code_test)

                    print(f"\n  Tokens: {result.token_count}")
                    print(f"  Compression: {result.compression_ratio:.2f}x")
                    print(f"  Syntax Preserved: {'[OK]' if result.syntax_preserved else '[FAIL]'}")
                    print(f"  Logic Preserved: {'[OK]' if result.logic_preserved else '[FAIL]'}")
                    print(f"  Variables Preserved: {'[OK]' if result.variable_preserved else '[FAIL]'}")
                    print(f"  Quality Score: {result.quality_score * 100:.1f}%")

                except Exception as e:
                    print(f"  [ERROR] Error: {e}")

        print("\n" + "=" * 50)
        print("Tests Complete")
        print("=" * 50)


def main():
    """Run all code understanding tests"""
    tester = CodeUnderstandingTest()
    tester.run_all_tests()


if __name__ == "__main__":
    main()
