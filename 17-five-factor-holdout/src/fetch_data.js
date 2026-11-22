// Refresh data/raw/ from the official Ken French Data Library.
// Run in the browser console on https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
// (the cloud build environment of this project cannot reach that host). For each zip it keeps the first
// table (for portfolios: "Average Value Weighted Returns -- Monthly"), the header line that names the
// CRSP vintage, and the SHA-256 of the zip, then downloads one CSV per file.
// Committed files: CRSP vintage 202609. SHA-256 prefixes of the zips used:
//   F-F_Research_Data_5_Factors_2x3 17d7a575e71d6764   F-F_Momentum_Factor 27ed60e5b2caa9ea
//   25_Portfolios_5x5 ec3a80f94edf98b7   25_Portfolios_ME_OP_5x5 8315899edcbd6497
//   25_Portfolios_ME_INV_5x5 b9743e512cbd8daa   10_Portfolios_Prior_12_2 688467caa28578a8
//   49_Industry_Portfolios d1a3b0b9896e1e63   F-F_Research_Data_Factors b03d39e4bb253dda
async function unzip(url) {
  const raw = await (await fetch(url)).arrayBuffer();
  const b = new Uint8Array(raw), dv = new DataView(raw);
  const method = dv.getUint16(8, true);
  let csize = dv.getUint32(18, true);
  const start = 30 + dv.getUint16(26, true) + dv.getUint16(28, true);
  if (csize === 0) { const cdo = dv.getUint32(b.length - 22 + 16, true); csize = dv.getUint32(cdo + 20, true); }
  const data = b.slice(start, start + csize);
  const sha = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', raw)))
    .map(x => x.toString(16).padStart(2, '0')).join('');
  if (method === 0) return [new TextDecoder().decode(data), sha];
  const s = new Blob([data]).stream().pipeThrough(new DecompressionStream('deflate-raw'));
  return [await new Response(s).text(), sha];
}
function firstTable(txt) {
  const L = txt.split(/\r?\n/);
  const i = L.findIndex(l => /^\s*,/.test(l));
  const out = ['#' + L.slice(0, i).join(' ').trim(), 'date' + L[i].replace(/\s+/g, '')];
  for (let j = i + 1; j < L.length; j++) {
    const l = L[j].trim();
    if (!/^\d{6}\s*,/.test(l)) break;
    out.push(l.replace(/\s+/g, ''));
  }
  return out.join('\n') + '\n';
}
const base = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/';
for (const f of ['F-F_Research_Data_5_Factors_2x3', 'F-F_Momentum_Factor', '25_Portfolios_5x5',
                 '25_Portfolios_ME_OP_5x5', '25_Portfolios_ME_INV_5x5', '10_Portfolios_Prior_12_2',
                 '49_Industry_Portfolios', 'F-F_Research_Data_Factors']) {
  const [txt, sha] = await unzip(base + f + '_CSV.zip');
  console.log(f, sha.slice(0, 16));
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([firstTable(txt)], {type: 'text/csv'}));
  a.download = f + '.csv'; a.click();
}
