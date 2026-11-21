// Download helper for the Ken French files used in project 05 (run in a browser console on
// https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html; the cloud build
// environment cannot reach that host). It unzips each file, keeps the first section
// ("Average Value Weighted Returns -- Monthly"; for the factors file, the monthly table) and saves CSVs.
async function unzip(url) {
  const b = new Uint8Array(await (await fetch(url)).arrayBuffer());
  const dv = new DataView(b.buffer);
  const method = dv.getUint16(8, true);
  let csize = dv.getUint32(18, true);
  const start = 30 + dv.getUint16(26, true) + dv.getUint16(28, true);
  if (csize === 0) { const cdo = dv.getUint32(b.length - 22 + 16, true); csize = dv.getUint32(cdo + 20, true); }
  const data = b.slice(start, start + csize);
  if (method === 0) return new TextDecoder().decode(data);
  const s = new Blob([data]).stream().pipeThrough(new DecompressionStream('deflate-raw'));
  return await new Response(s).text();
}
function firstMonthlyTable(txt, headerRe) {
  const lines = txt.split(/\r?\n/);
  const i = lines.findIndex(l => headerRe.test(l));
  const out = ['date' + lines[i].replace(/\s+/g, '')];
  for (let j = i + 1; j < lines.length; j++) {
    const l = lines[j].trim();
    if (!/^\d{6},/.test(l)) break;
    const [d, ...v] = l.split(',');
    const end = new Date(Date.UTC(+d.slice(0, 4), +d.slice(4), 0)).toISOString().slice(0, 10);
    out.push([end, ...v.map(x => (parseFloat(x) / 100).toFixed(4))].join(','));
  }
  return out.join('\n');
}
const base = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/';
for (const [zip, out, re] of [
  ['12_Industry_Portfolios_CSV.zip', 'kf_12_industries_vw_monthly.csv', /^\s*,NoDur/],
  ['49_Industry_Portfolios_CSV.zip', 'kf_49_industries_vw_monthly.csv', /^\s*,Agric/],
  ['F-F_Research_Data_Factors_CSV.zip', 'kf_factors_monthly.csv', /^\s*,Mkt-RF/]]) {
  const csv = firstMonthlyTable(await unzip(base + zip), re);
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([csv], {type: 'text/csv'})); a.download = out; a.click();
}
// Missing values (-99.99) appear in the 49-industry file before 1969-07; src/data.py starts there.
