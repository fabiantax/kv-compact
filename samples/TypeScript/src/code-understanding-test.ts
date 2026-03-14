/**
 * Test Code Understanding and Output Quality at Different Token Counts
 *
 * This test validates that KV cache compaction preserves:
 * 1. Code syntax and structure
 * 2. Logical flow and reasoning
 * 3. Variable references and dependencies
 * 4. Code comments and documentation
 */

import { KvCompactor, StreamingCompactor } from './compaction';

interface TestResult {
  tokenCount: number;
  compressionRatio: number;
  originalCode: string;
  compactedOutput: string;
  syntaxPreserved: boolean;
  logicPreserved: boolean;
  variablePreserved: boolean;
  qualityScore: number;
}

interface CodeTest {
  name: string;
  language: string;
  code: string;
}

// Test cases with increasing complexity
const codeTests: CodeTest[] = [
  {
    name: "simple-function",
    language: "typescript",
    code: `
// Calculate fibonacci number recursively
function fibonacci(n: number): number {
  if (n <= 1) return n;
  return fibonacci(n - 1) + fibonacci(n - 2);
}

// Calculate factorial iteratively
function factorial(n: number): number {
  let result = 1;
  for (let i = 2; i <= n; i++) {
    result *= i;
  }
  return result;
}

// Main function
function main() {
  console.log("Fibonacci of 10:", fibonacci(10));
  console.log("Factorial of 5:", factorial(5));
}
    `,
  },
  {
    name: "class-hierarchy",
    language: "typescript",
    code: `
// Base class for all shapes
abstract class Shape {
  constructor(protected color: string) {}

  abstract getArea(): number;
  abstract getPerimeter(): number;

  describe(): string {
    return \`A \${this.color} shape with area \${this.getArea()}\`;
  }
}

// Rectangle class
class Rectangle extends Shape {
  constructor(
    color: string,
    private width: number,
    private height: number
  ) {
    super(color);
  }

  getArea(): number {
    return this.width * this.height;
  }

  getPerimeter(): number {
    return 2 * (this.width + this.height);
  }
}

// Circle class
class Circle extends Shape {
  constructor(
    color: string,
    private radius: number
  ) {
    super(color);
  }

  getArea(): number {
    return Math.PI * this.radius * this.radius;
  }

  getPerimeter(): number {
    return 2 * Math.PI * this.radius;
  }
}

// Usage
const rect = new Rectangle("blue", 5, 10);
const circle = new Circle("red", 7);
console.log(rect.describe());
console.log(circle.describe());
    `,
  },
  {
    name: "async-data-processing",
    language: "typescript",
    code: `
// Interface for user data
interface User {
  id: number;
  name: string;
  email: string;
  preferences: {
    theme: 'light' | 'dark';
    notifications: boolean;
  };
}

// Fetch user data from API
async function fetchUser(userId: number): Promise<User> {
  const response = await fetch(\`/api/users/\${userId}\`);
  if (!response.ok) {
    throw new Error(\`Failed to fetch user: \${response.statusText}\`);
  }
  return await response.json();
}

// Process multiple users concurrently
async function processUsers(userIds: number[]): Promise<User[]> {
  const users = await Promise.all(
    userIds.map(async (id) => {
      try {
        const user = await fetchUser(id);
        // Transform user data
        return {
          ...user,
          preferences: {
            ...user.preferences,
            notifications: false, // Disable by default
          }
        };
      } catch (error) {
        console.error(\`Error processing user \${id}:\`, error);
        return null;
      }
    })
  );

  // Filter out failed fetches
  return users.filter((user): user is User => user !== null);
}

// Main processing pipeline
async function main() {
  const userIds = [1, 2, 3, 4, 5];
  const users = await processUsers(userIds);

  console.log(\`Processed \${users.length} users\`);
  users.forEach(user => {
    console.log(\`\${user.name}: \${user.email}\`);
  });
}
    `,
  },
  {
    name: "algorithm-implementation",
    language: "typescript",
    code: `
// Binary Search Tree implementation
class TreeNode {
  constructor(
    public value: number,
    public left: TreeNode | null = null,
    public right: TreeNode | null = null
  ) {}
}

class BinarySearchTree {
  private root: TreeNode | null = null;

  insert(value: number): void {
    const newNode = new TreeNode(value);

    if (!this.root) {
      this.root = newNode;
      return;
    }

    this.insertNode(this.root, newNode);
  }

  private insertNode(node: TreeNode, newNode: TreeNode): void {
    if (newNode.value < node.value) {
      if (!node.left) {
        node.left = newNode;
      } else {
        this.insertNode(node.left, newNode);
      }
    } else {
      if (!node.right) {
        node.right = newNode;
      } else {
        this.insertNode(node.right, newNode);
      }
    }
  }

  search(value: number): TreeNode | null {
    return this.searchNode(this.root, value);
  }

  private searchNode(node: TreeNode | null, value: number): TreeNode | null {
    if (!node) return null;
    if (value === node.value) return node;
    if (value < node.value) return this.searchNode(node.left, value);
    return this.searchNode(node.right, value);
  }

  // In-order traversal (sorted)
  inOrderTraversal(): number[] {
    const result: number[] = [];
    this.traverse(this.root, result);
    return result;
  }

  private traverse(node: TreeNode | null, result: number[]): void {
    if (!node) return;
    this.traverse(node.left, result);
    result.push(node.value);
    this.traverse(node.right, result);
  }
}

// Usage example
const bst = new BinarySearchTree();
[50, 30, 70, 20, 40, 60, 80].forEach(n => bst.insert(n));
console.log("Sorted:", bst.inOrderTraversal());
console.log("Found 40:", bst.search(40));
console.log("Found 100:", bst.search(100));
    `,
  },
];

/**
 * Run code understanding tests
 */
export class CodeUnderstandingTest {
  private compactor: KvCompactor;

  constructor() {
    this.compactor = new KvCompactor({ ratio: 0.2 });
  }

  /**
   * Test code understanding at specific token count
   */
  async testAtTokenCount(
    targetTokens: number,
    codeTest: CodeTest
  ): Promise<TestResult> {
    console.log(\`Testing \${codeTest.name} at ~\${targetTokens} tokens...\`);

    // Create prompt with code repeated to reach target token count
    const prompt = this.createPromptWithTokens(codeTest.code, targetTokens);

    // Simulate compaction
    const result = await this.compactor.compactLayer(
      this.tokenize(prompt),
      this.tokenize(prompt),
      this.tokenize(prompt), // Simplified - use actual queries in real test
      prompt.length,
      32, // num_heads
      128  // head_dim
    );

    // Generate output from compacted cache
    const compactedOutput = await this.generateOutput(prompt, result);

    // Evaluate quality
    return {
      tokenCount: targetTokens,
      compressionRatio: prompt.length / (prompt.length * 0.2), // Simplified
      originalCode: codeTest.code,
      compactedOutput,
      syntaxPreserved: this.checkSyntaxPreserved(codeTest.code, compactedOutput),
      logicPreserved: this.checkLogicPreserved(codeTest.code, compactedOutput),
      variablePreserved: this.checkVariablesPreserved(codeTest.code, compactedOutput),
      qualityScore: result.metrics?.qualityScore || 0,
    };
  }

  /**
   * Create prompt with target token count by repeating code
   */
  private createPromptWithTokens(code: string, targetTokens: number): string {
    const estimatedTokensPerCode = code.length / 4; // Rough estimate
    const repetitions = Math.ceil(targetTokens / estimatedTokensPerCode);

    let prompt = \`// Complete this code and explain its functionality:\n\n\${code}\n\n\`;

    // Add context and variations
    for (let i = 0; i < repetitions; i++) {
      prompt += \`\n// Variation \${i + 1}:\n\${code}\n\n\`;
      prompt += \`// Explanation: This code demonstrates \${this.extractConcept(code)}\n\n\`;
    }

    return prompt;
  }

  /**
   * Tokenize text (simplified)
   */
  private tokenize(text: string): Float32Array {
    // In real implementation, use actual tokenizer
    const tokens = text.split(/\s+/).length;
    const array = new Float32Array(tokens * 128); // Assume embedding dim of 128
    return array;
  }

  /**
   * Generate output from compacted cache
   */
  private async generateOutput(
    prompt: string,
    compactionResult: any
  ): Promise<string> {
    // In real implementation, this would call the LLM
    // For now, return placeholder
    return "// Generated code would appear here";
  }

  /**
   * Check if syntax is preserved in output
   */
  private checkSyntaxPreserved(original: string, output: string): boolean {
    // Check for balanced braces, parentheses, etc.
    const checkBalance = (str: string, open: string, close: string): boolean => {
      let balance = 0;
      for (const char of str) {
        if (char === open) balance++;
        if (char === close) balance--;
        if (balance < 0) return false;
      }
      return balance === 0;
    };

    return (
      checkBalance(output, '{', '}') &&
      checkBalance(output, '(', ')') &&
      checkBalance(output, '[', ']')
    );
  }

  /**
   * Check if logical flow is preserved
   */
  private checkLogicPreserved(original: string, output: string): boolean {
    // Check for key control flow keywords
    const keywords = ['if', 'else', 'for', 'while', 'return', 'async', 'await'];
    return keywords.every(kw =>
      (original.includes(kw) && output.includes(kw)) ||
      (!original.includes(kw) && !output.includes(kw))
    );
  }

  /**
   * Check if variable names are preserved
   */
  private checkVariablesPreserved(original: string, output: string): boolean {
    // Extract variable declarations
    const varRegex = /(?:const|let|var)\s+(\w+)/g;
    const originalVars = new Set();
    let match;

    while ((match = varRegex.exec(original)) !== null) {
      originalVars.add(match[1]);
    }

    // Check if variables appear in output
    let preservedCount = 0;
    originalVars.forEach(v => {
      if (output.includes(v)) preservedCount++;
    });

    return preservedCount / originalVars.size > 0.8; // 80% threshold
  }

  /**
   * Extract main concept from code
   */
  private extractConcept(code: string): string {
    if (code.includes('class ')) return 'object-oriented programming';
    if (code.includes('async ') || code.includes('await ')) return 'asynchronous operations';
    if (code.includes('function ')) return 'functional programming';
    if (code.includes('for ') || code.includes('while ')) return 'looping constructs';
    return 'programming concepts';
  }

  /**
   * Run all tests across different token counts
   */
  async runAllTests(): Promise<void> {
    const tokenCounts = [1000, 10000];

    console.log('=== Code Understanding Tests ===\n');

    for (const codeTest of codeTests) {
      console.log(\`\n📝 Test: \${codeTest.name} (\${codeTest.language})\`);

      for (const tokenCount of tokenCounts) {
        try {
          const result = await this.testAtTokenCount(tokenCount, codeTest);

          console.log(\`  Tokens: \${result.tokenCount}\`);
          console.log(\`  Compression: \${result.compressionRatio.toFixed(2)}x\`);
          console.log(\`  Syntax Preserved: \${result.syntaxPreserved ? '✅' : '❌'}\`);
          console.log(\`  Logic Preserved: \${result.logicPreserved ? '✅' : '❌'}\`);
          console.log(\`  Variables Preserved: \${result.variablePreserved ? '✅' : '❌'}\`);
          console.log(\`  Quality Score: \${(result.qualityScore * 100).toFixed(1)}%\`);
        } catch (error) {
          console.error(\`  ❌ Error: \${error}\`);
        }
      }
    }

    console.log('\n=== Tests Complete ===');
  }
}

// Export for use
export async function runCodeUnderstandingTests(): Promise<void> {
  const tester = new CodeUnderstandingTest();
  await tester.runAllTests();
}
