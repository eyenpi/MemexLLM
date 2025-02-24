import React, {ReactElement} from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';

export default function Home(): ReactElement {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={siteConfig.title}
      description="MemexLLM - LLM Conversation Management">
      <main style={{
        padding: '4rem 0',
        textAlign: 'center',
        maxWidth: '800px',
        margin: '0 auto',
      }}>
        <h1>{siteConfig.title}</h1>
        <p style={{fontSize: '1.2rem', margin: '2rem 0'}}>
          {siteConfig.tagline}
        </p>
        <div style={{display: 'flex', gap: '1rem', justifyContent: 'center'}}>
          <Link
            className="button button--primary button--lg"
            to="/docs/intro">
            Documentation
          </Link>
          <Link
            className="button button--secondary button--lg"
            href="https://github.com/eyenpi/MemexLLM">
            GitHub
          </Link>
        </div>
      </main>
    </Layout>
  );
}
