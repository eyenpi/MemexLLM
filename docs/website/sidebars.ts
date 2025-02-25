import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  docs: [
    'intro',
    {
      type: 'category',
      label: 'Integrations',
      items: [
        'integrations/overview',
        'integrations/openai',
      ],
    },
    {
      type: 'category',
      label: 'Storage',
      items: [
        'storage/overview',
        'storage/memory',
        'storage/sqlite',
        'storage/custom',
      ],
    },
    {
      type: 'category',
      label: 'Algorithms',
      items: [
        'algorithms/overview',
        'algorithms/fifo',
        'algorithms/custom',
      ],
    },
    {
      type: 'category',
      label: 'Examples',
      items: [
        'examples/simple_chatbot',
        'examples/custom_context',
      ],
    },
  ],
};

export default sidebars;
