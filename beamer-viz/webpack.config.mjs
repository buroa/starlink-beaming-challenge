import CopyPlugin from 'copy-webpack-plugin';
import HtmlPlugin from 'html-webpack-plugin';
import { dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

export default {
  mode: 'production',
  entry: './index.js',
  output: {
    path: __dirname + '/dist',
    // Content-hashed so a fresh deploy never serves a stale bundle from cache
    // (wasm-pack's .wasm output is already content-hashed by webpack).
    filename: '[name].[contenthash].js',
    chunkFilename: '[name].[contenthash].js',
    clean: true // wipe prior builds' hashed bundles
  },
  plugins: [
    // Emits dist/index.html from the template with the hashed entry <script>
    // injected at the end of <body> (after the coi-serviceworker in <head>).
    new HtmlPlugin({ template: 'index.html', inject: 'body' }),
    new CopyPlugin({
      patterns: [
        '../coi-serviceworker.js',
        { from: '../test_cases', to: 'test_cases' }
      ]
    })
  ],
  module: {
    rules: [
      {
        test: /\.m?js$/,
        resolve: { fullySpecified: false }
      }
    ]
  }
};
