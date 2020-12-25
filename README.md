# cses-sol-hard

**Traffic Lights**

	#include <bits/stdc++.h>
	using namespace std;
	#define z insert(
	multiset<int> k,p;
	int N,M;
	main(){
			cin>>N>>M;
			p.z 0);
			p.z N);
			k.z N);
			for(;cin>>N;){
			  auto y=p.lower_bound(N),x=--y;
			  k.erase(k.find(*++y-*x));
			  k.z N-*x);
			  k.z *y-N);
			  auto s=k.end();
			  cout<<*--s<<" ";
			  p.z N);
			}
	}
  
**Sliding Cost ** 

	#include <bits/stdc++.h>
	#define ll long long
	using namespace std;
	int n, k, old_mid;
	int main()
	{
		cin >> n >> k;
		vector<int> v(n);
		for (auto &x : v)
			cin >> x;
		multiset<int> id(v.begin(), v.begin() + k);
		auto mid = next(id.begin(), (k + 1) / 2 - 1);
		ll d = 0;
		for (int i = 0; i < k; i++)
			d += abs(*mid - v[i]);
		cout << d << " ";
		old_mid = *mid;
		for (int i = k; i < n; i++)
		{
			id.insert(v[i]);
			if (v[i] < *mid)
				mid--;
			if (v[i - k] <= *mid)
				mid++;
			id.erase(id.find(v[i - k]));
			d += abs(v[i] - *mid) - abs(v[i - k] - old_mid);
			if (k % 2 == 0)
				d -= (*mid - old_mid);
			old_mid = *mid;
			cout << d << " ";
		}
	}

**Maximum Subarray Sum II ** 

	#include<bits/stdc++.h>
	using namespace std;
	using l=long long;
	main()
	{ 
		l n,a,b,y[1+(l)2e5],z=-1e18,j=1;
		cin>>n>>a>>b;
		set<array<l,2>> x;
		for(l i=0;i<n;i++,j++){
			cin>>y[i+1];
			y[i+1]+=y[i];
			if(j>=a)x.insert({y[j-a],j-a});
			if(x.size())z=max(z,y[j]-(*x.begin())[0]);
			if(j>=b)x.erase({y[j-b],j-b});
		}
		cout<<z;
	}


**Round Trip ** 

	#include <bits/stdc++.h>
	using namespace std;

	#define i int
	#define M '   '
	#define b push_back

	i p[M], x, y, k;
	vector<i> g[M], s;

	i f(i w, i r) {
		p[w] = r;

		for (i u: g[w])
			if (u - r) { 
				if (p[u]) {
					s.b(u), s.b(w);
					while (w - u)
						w = p[w], s.b(w);

					cout << size(s);
					for (i j: s) cout << ' ' << j;
					exit(0);
				}
				f(u, w);    
			}
	}
 
	main() {
		cin >> x >> y;
		while (cin >> x >> y)
			g[x].b(y), g[y].b(x);

		while (++k < M)
			if (!p[k]) 
				f(k, M);

		cout << "IMPOSSIBLE";
	}

**Planets Queries I** 

	#include <iostream>
	#define f(_,x,y) for(_=x;_<=y;_++)

	using namespace std;

	int d[200005][31], n, q, i, j, k;

	main () {
	  cin >> n >> q;
	  f(i,1,n) cin >> d[i][0];
	  f(j,1,30) f(i,1,n) d[i][j] = d[d[i][j-1]][j-1];
	  while (q--) {
		cin >> n >> k;
		f(i,0,30)
		  if (k&(1<<i)) n = d[n][i];
		cout << n << '\n';
	  }
	}

**Planets Queries II **

	#include <iostream>
	using std::cin;

	const int N = 3e5;
	int d[N], y[N], c[N], g[18][N], p[N], n, m, i, j, k, u, v, *t = *g;
	#define f for(
	#define U [u]
	#define V [v]

	main() {
		cin >> n >> m;
		f ;i<n;) cin >> t[++i];

		f ++i;--i;) if(!c[i]) {
			d[p[k=0] = i] = N;
			f u = t[i]; !d U; u = t U) d[p[++k] = u] = N;
			j = d U == N;
			f v=p[k]; k>=0; v=p[--k])
				f
					c V = c[t V] + 1,
					j ? y V = u, t V = 0 : 0,
					d V = d[t V] + 1,
					j &= v != u,
					n=0; n<17; n++
				) g[n+1] V = g[n][g[n] V];
		}

		f ;cin >> u >> v;) {
			k = d U - d V;
			f n=0; n<18; n++) k & 1<<n ? u = g[n] U : 0;
			printf("%d\n", y V && y U == y V ? k + (c U - c V + c[y V]) % c[y V] : u^v ? -1 : k);
		}
	}

**Planets Cycles ** 

	#include <bits/stdc++.h>
	#define rep(i,a,b) for(int i = a; i <= b; ++i)
	using namespace std;
	const int N = 2e5+5;
	int p[N],in[N],c[N],d[N],t=0;

	void dfs(int u){
		if(in[u] == 0){
			in[u]=++t;
			int v=p[u];
			if(in[v] > 0){
				c[u] = in[u] - in[v] + 1;
				d[u] = 1;
			}
			else if(in[v] < 0){
				d[u] = max(c[v],d[v]) + 1;
			}
			else{
				dfs(v);
				d[u] = d[v] + 1;
				c[u] = c[v]; 
			}
			in[u]*=-1;
		}
	}

	int main(){
		int n;
		scanf("%d",&n);
		rep(i,1,n)scanf("%d",&p[i]);
		rep(i,1,n)dfs(i);
		rep(i,1,n)printf("%d ",max(c[i],d[i]));
		printf("\n");
	}

**Giant Pizza **

	#include <bits/stdc++.h>
	using namespace std;
	#define p push_back
	#define N 200005

	stack <int> s;
	vector <int> v[N], w[N];
	int c[N], sc, n, m, i, o, u;
	bool vs[N], vs2[N];

	void f(int x) {
		vs[x] = 1;
		for (auto y : v[x])
			if (!vs[y])
				f(y);
		s.push(x);
	}

	#define g(x) x>m?x-m:x+m
	void f2(int x) {
		vs2[x] = 1;
		if (c[g(x)] == (c[x] = sc)) cout << "IMPOSSIBLE", exit(0);
		for (auto y : w[x])
			if (!vs2[y])
				f2(y);
	}

	#define d(x,y) v[g(x)].p(y);v[g(y)].p(x);w[y].p(g(x));w[x].p(g(y))
	int main() {
		cin >> n >> m;
		while (n--) {
			char a, b;
			cin >> a >> o >> b >> u;
			d(o+m*(a>'+'),u+m*(b>'+'));
		}
		for (i = 1; i <= m + m; i++)
			if (!vs[i])
				f(i);
		while (s.size()) {
			i = s.top();
			s.pop();
			if (!vs2[i])
				++sc, f2(i);
		}
		for (i = 1; i <= m; i++)
			cout << (c[i]>c[m+i]?"+ ":"- ");
		return 0;
	}

**Mail Delivery** 

	#include <bits/stdc++.h>
	using namespace std;
	set<int>g[200005];
	vector<int>v;

	void dfs(int c)
	{
		while(!g[c].empty())
		{
			int u = *g[c].begin();
			g[c].erase(u);
			g[u].erase(c);
			dfs(u);
		}
		v.push_back(c);
	}

	int main()
	{

		int n,m;
		cin>>n>>m;
		for(int i=1; i<=m; i++)
		{
			int u,v;
			cin>>u>>v;
			g[u].insert(v);
			g[v].insert(u);
		}

		for(int i=1; i<=n; i++)
			if(g[i].size()&1)
				return cout<<"IMPOSSIBLE",0;

		dfs(1);

		if(v.size() != m+1)
			return cout<<"IMPOSSIBLE",0;

		for(int&x:v)
		cout<<x<<" ";
	}

**De Bruijn Sequence** 

	#include <iostream>
	using namespace std;
	int main()
	{
		int n;
		cin >> n;
		int k;
		k = (1 << n);
		bool ok[k];
		for(int i = 0; i < k; i++) ok[i] = 0;
		for(int i = 0; i < n; i++) cout << 0;
		ok[0] = 1;
		int a = 0;
		for(int i = 0; i < k-1; i++){
			a = a % (k/2);
			a <<= 1;
			if(ok[a+1] == 0){
				cout << 1;
				a++;
			}
			else cout << 0;
			ok[a] = 1;
		}
		return 0;
	}

**Hamiltonian Flights** 

	#include <bits/stdc++.h>
	using namespace std;
	#define q int
	const q MOD = 1e9+7;
	q dp[20][1048576];
	q main()
	{
		q n, m;
		cin >> n >> m;
		vector<vector<int>> g(n);
		while (m--)
		{
			q a, b;
			cin >> a >> b;
			a--; b--;
			g[a].push_back(b);
		}

		dp[0][1] = 1;
		q r = 1 << n;
		for (q a = 1; a < r; a++)
		{
			for (q i = 0; i < n; i++)
			{
				if (1 << i & a && dp[i][a] != 0)
				{
					for (q j : g[i])
					{
						dp[j][a ^ (1 << j)] += dp[i][a];
						dp[j][a ^ (1 << j)] %= MOD;
					}
				}
			}
		}
		cout << dp[n - 1][r - 1] << "\n";
	}

**Download Speed**

	#include <bits/stdc++.h>
	using namespace std;
	#define ll long long
	ll g[500][500];
	int main()
	{
		int n, m; cin >> n >> m;
		ll f = 0;
		while (m--)
		{
			ll x, y, c; cin >> x >> y >> c;
			g[x-1][y-1] += c;
		}
		while (1)
		{
			queue<ll> q;
			vector<bool> w(n);
			vector<int> p(n, -1);
			q.push(0);
			w[0] = 1;
			ll mf = 1e18;
			while (q.size() && !w[n-1])
			{
				int u = q.front(); q.pop();
				for (int v = 0; v < n; v++)
					if (g[u][v] && !w[v])
						{
							w[v] = 1;
							p[v] = u;
							q.push(v);
						}
			}
			if (q.empty()) break;
			vector<int> t;
			for (int v = n-1; v != -1; v = p[v])
			{
				t.push_back(v);
				if (v) mf = min(mf, g[p[v]][v]);
			}
			for (int v = t.size()-1; v >= 1; v--)
			{
				g[t[v]][t[v-1]] -= mf;
				g[t[v-1]][t[v]] += mf;
			}
			f += mf;
		}
		cout << f;
	}
  
**Police Chase ** 

	#include <bits/stdc++.h>
	#define rep(i,a,b)  for (int i=a; i<(b); ++i)
	using namespace std;

	int main() {
		cin.tie(0);
		ios_base::sync_with_stdio(0);

		int n,m;
		cin>>n>>m;
		vector<vector<int>> g(n,vector<int>(n));
		rep(i,0,m) {
			int a,b;
			cin>>a>>b;
			a--,b--;
			g[a][b]=g[b][a]=1;
		}

		auto c=g;
		vector<int> w(n);
		function<bool(int,int,int)> aug=[&](int v,int t,int W) {
			if (v==t) return 1;
			if (w[v]==W) return 0;
			w[v]=W;
			rep(u,0,n) if (c[v][u]>0 && aug(u,t,W)) {
				c[v][u]--;
				c[u][v]++;
				return 1;
			}
			return 0;
		};

		int W=0,R=0;
		while (aug(0,n-1,++W)) R++;

		cout<<R<<'\n';
		rep(i,0,n) if (w[i]==W) {
			rep(j,0,n) if (w[j]!=W && g[i][j] && !c[i][j]) cout<<i+1<<" "<<j+1<<'\n';
		}
	}
**Distinct Routes ** 

	#include<bits/stdc++.h>
	#define pb emplace_back
	using namespace std;
	const int N = 505, M = 4005;
	int n,m,a,b,ans,ev[M],val[M];
	vector<int> g[N],gp[N],v,t;
	bitset<M> vis;

	int dfs(int u){
		vis[u] = 1;
		if(u == n){
			ans++;
			return 1;
		}
		for(int i : g[u]){
			if(vis[ev[i]] or val[i] == 0) continue;
			if(dfs(ev[i])){
				val[i] = 0;
				val[i^1] = 1;
				return 1;
			}
		}
		return 0;
	}

	signed main(void){
		cin >> n >> m;
		for(int i = 0 ; i < 2*m; i += 2){
			cin >> a >> b;
			ev[i] = b;
			ev[i^1] = a;
			val[i] = 1;
			g[a].pb(i);
			g[b].pb(i^1);
		}

		while(dfs(1)) vis = 0 ;

		cout << ans << '\n';
		vis = 0;

		for(int i = 0; i < ans; i++){
			v.clear();
			int u = 1;
			v.pb(u);
			while(u != n){
				 for(int j : g[u]){
					if(!(j&1) && !val[j]){
						u = ev[j];
						val[j] = 1;
						v.pb(u);
						break;
					}
				}
			}
			cout << v.size() << '\n';
			for(int j : v)
				cout << j << ' ';
			cout << '\n';
		}

	}
	
**Range Update Queries ** 

	#include <bits/stdc++.h>
	using namespace std;
	using ll = long long;
	int n;
	ll t[300005];
	void f(int k, ll x) {
	  while (k <= n) t[k] += x, k += k & -k;
	}

	void f(int a, int b, ll x) {
	  f(a, x), f(b + 1, -x);
	}

	ll g(int k) {
	  ll s = 0;
	  while (k) s += t[k], k -= k & -k;
	  return s;
	}

	int main() {
	  int q, p, x, y = 0, z;
	  cin >> n >> q;
	  for (int i = 1; i <= n; i++) cin >> x, f(i, x - y), y = x;
	  while (q--) {
		cin >> p;
		if (p == 1) cin >> x >> y >> z, f(x, y, z);
		else cin >> x, cout << g(x) << "\n";
	  }
	}
**Hotel Queries ** 

	#include <iostream> 
	using namespace std;
	#define f() for (i = y = 0; k = i / P, i < n; ++i)
	main(){
		int P = 450, n, m, x, y, i, k, a[P*P], b[P];
		cin >> n >> m;
		f() cin >> a[i], b[k] = 2e9;
		for(;m--, cin >> x; cout << y << " ")
			f() i%P || !y * b[k] >= x ? y | a[i] < x ? : (a[y = i] -= x, ++y),
			b[k] = max(i%P ? b[k] : 0, a[i]) : i += P - 1;
	}

**Subarray Sum Queries ** 

	#include <bits/stdc++.h> 
	using namespace std;
	#define m (l + r) / 2
	#define L v * 2
	#define R L + 1
	#define A int

	const A N = 800000;

	struct D{
		int64_t l, r, s, b;
		D(){}
		D(A y){
			l = r = b = max(y, 0);
			s = y;
		}
		D M(D a){
			a.b = max({b, a.b, r + a.l});
			a.l = max(l, s + a.l),
			a.r = max(a.r, a.s + r),
			a.s += s;
			return a;
		}
	};

	D t[N];
	A a[N], x, y, n, q;

	void B(A v, A l, A r){
		l == r - 1 ? t[v] = {a[l]} : (B(L, l, m), B(R, m, r), t[v] = t[L].M(t[R]));
	}

	void U(A v, A l, A r, A p, A x){
		l == r - 1 ? t[v] = {x} : ((p < m ? U(L, l, m, p, x) : U(R, m, r, p, x)), t[v] = t[L].M(t[R]), D());
	}

	main(){
		cin >> n >> q;
		for (x = 0; x < n; ++x) cin >> a[x];
		B(1, 0, n);
		while (q--){
			cin >> x >> y;
			U(1, 0, n, x - 1, y);
			cout << t[1].b << "\n";
		}
	}

**Distinct Values Queries ** 

	#include <bits/stdc++.h>
	using namespace std;
	const int N=200010;
	int n,h[N],q,la[N],fen[N],ans[N],i;
	set<int> s;
	map<int,int> g;
	vector<pair<int,int>> x[N];

	int que(int i)
	{
		int s=0;
		for (;i;i-=i&-i) s+=fen[i];
		return s;
	}

	main()
	{
		cin>>n>>q;
		for (int i=0;i<n;++i) cin>>h[i],s.insert(h[i]);
		for (auto a: s) g[a]=i++;
		for (int i=0;i<n;++i)
		{
			h[i]=g[h[i]];
			int a=h[i];
			h[i]=la[a]+1;
			la[a]=i+1;
		}
		for (int i=0;i<q;++i)
		{
			int a,b;
			cin>>a>>b;
			x[b-1].push_back({i,a});
		}
		for (int i=0;i<n;++i)
		{
			int s=h[i];
			for (;s<=n;s+=s&-s) ++fen[s];
			for (auto w: x[i])
			{
				ans[w.first]=que(w.second)-w.second+1;
			}
		}
		for (int i=0;i<q;++i) cout<<ans[i]<<'\n';
	}

**Forest Queries II ** 

	#include <bits/stdc++.h>
	using namespace std;
	const int N = 1001;
	int F[N][N] = {};
	bool A[N][N] = {};
	int n, m;

	int q(int i, int j){
		int r = 0, k;
		while(j){
			k = i;
			while(k){
				r += F[j][k];
				k -= k & (-k);
			}
			j -= j & (-j);
		}
		return r;
	}

	void u(int i, int j){
		int k, d = A[i][j] ? -1 : 1;
		A[i][j] = !A[i][j];
		i++; j++;
		while(j <= n){
			k = i;
			while(k <= n){
				F[j][k] += d;
				k += k & (-k);
			}
			j += j & (-j);
		}
	}

	int main(){
		cin >> n >> m;
		char ch;
		for (int i = 0; i < n; i++){
			for (int j = 0; j < n; j++){
				cin >> ch;
				if (ch == '*') u(i, j);
			}
		}

		int t, a, b, c, d;
		while(m--){
			cin >> t >> a >> b; a--; b--;
			if (t == 1) u(a, b);
			else{
				cin >> c >> d;
				cout << q(c, d) + q(a, b) - q(a, d) - q(c, b) << '\n';
			}
		}
	}

**Range Updates and Sums ** 

	#include <bits/stdc++.h> 
	using namespace std;
	const int maxn = 2e5+11;
	typedef long long ll;
	ll t[4*maxn+7];
	ll k[4*maxn+7];
	bool j[4*maxn+7];
	int n;

	void p(int h, int tl, int tr){
		if (tl == tr) {
			if (j[h]) t[h] = k[h];
			else t[h] += k[h];
		}
		else{
			if (j[h]){
				t[h] = k[h]*(tr-tl+1);
				k[h*2] = k[h];
				k[h*2+1] = k[h];
				j[h*2] = 1;
				j[h*2+1] = 1;
			}
			else{
				t[h] += k[h]*(tr-tl+1);
				k[h*2] += k[h];
				k[h*2+1] += k[h];
			}
		}
		k[h] = 0;
		j[h] = false;
	}

	ll u(int op, int l, int r, ll v, int h = 1, int tl = 0, int tr = n-1){
		p(h, tl, tr);
		if (l > r) return 0;
		if (tl == l && tr == r){
			if (op == 0) k[h] += v;
			else{
				k[h] = v;
				j[h] = 1;
			}
			p(h, tl, tr);
			return t[h];
		}
		int mid = (tl+tr)/2;
		ll g = u(op, l, min(r, mid), v, h*2, tl, mid);
		ll f = u(op, max(mid+1, l), r, v, h*2+1, mid+1, tr);
		t[h] = t[h*2]+t[h*2+1];
		return g+f;
	}
	int main(){
		int q;cin>>n>>q;
		for(int i = 0;i < n;i++){
			int x;cin>>x;
			u(1, i, i, x);
		}

		for(int i = 0;i < q;i++){
			int op;cin>>op;
			if (op == 1){
				int l, r, x;cin>>l>>r>>x;
				l--;r--;
				u(0, l, r, x);
			}
			else if (op == 2){
				int l, r, x;cin>>l>>r>>x;
				l--;r--;
				u(1, l, r, x);
			}
			else if (op == 3){
				int l, r;cin>>l>>r;
				l--;r--;
				cout << u(0, l, r, 0) << endl;
			}
		}
		return 0;
	}

**Polynomial Queries ** 

	#include <iostream>
	using namespace std;
	typedef long long ll;
	ll BIT[200001][3];
	ll query(int u) {
		ll mx = 0, rx = 0, dx = 0;
		for(int i = u; i; i -= i & (-i)) {
			mx += BIT[i][0];
			rx += BIT[i][1];
			dx += BIT[i][2];
		}
		mx += rx * u;
		mx += dx * u * (u + 1) / 2;       
		return mx;
	}

	void up(int u, int r, ll d) {
		while(u < 200002) {
			BIT[u][r] += d;
			u += u & (-u);
		}
		return ;
	}

	void upd(int v, ll i, ll j) {

		up(1, 2, i);
		up(v + 1, 2, -i);
		up(v + 1, 0, i * v * (v + 1) / 2);

		up(1, 1, j);
		up(v + 1, 1, -j);
		up(v + 1, 0, v * j);

		return ;
	}

	int main() {
		int n, q, u, v, h; cin >> n >> q;
		for(int i = 0; i < n; ++i) {
			cin >> u;
			for(int j = i + 1; j <= n; j += j & (-j))
				BIT[j][0] += u;
		}

		while(q--) {
			cin >> h >> u >> v;
			if(--h) cout << query(v) - query(u - 1) << '\n';
			else upd(v, 1, 1 - u), upd(u - 1, -1, u - 1);
		}
		return 0;
	}

**Range Queries and Copies ** 

	#include<bits/stdc++.h>
	#define f return
	#define p new o
	#define g (l+r)/2
	using w=int64_t;
	using namespace std;
	#define e cin
	w t[1<<18],n,m,z,a,b,c,i,s=1;
	struct o{
	  o *l,*r;
	  w v;
	  o(w x=0,o*l=0,o*r=0):l(l),r(r),v(x){}
	  void h(){v=l->v+r->v;}
	}*r[1<<18];
	o* d(w l,w r){
	  if(l==r)f p(t[l]);
	  o* k=p(0,d(l,g),d(g+1,r));
	  f k->h(),k;
	}
	o* u(w i,w x,o* j,w l,w r){
	  if(i<l||i>r)f j;
	  if(l==r)f p(x);
	  o* k=p(0,u(i,x,j->l,l,g),u(i,x,j->r,g+1,r));
	  f k->h(),k;
	}
	w q(w x,w y,o* j,w l,w r) {
	  if(x>r||y<l)f 0;
	  if(x<=l&&r<=y)f j->v;
	  f q(x,y,j->l,l,g)+q(x,y,j->r,g+1,r);
	}
	main(){
	  e>>n>>m;
	  for(;++i<=n;)e>>t[i];
	  r[1]=d(1,n);
	  while(m--){
		e>>z;
		if(z==1)e>>a>>b>>c,r[a]=u(b,c,r[a],1,n);
		if(z==2)e>>a>>b>>c,cout<<q(b,c,r[a],1,n)<<'\n';
		if(z==3)e>>a,r[++s]=r[a];
	  }
	}
	
**Counting Paths ** 

	#include <bits/stdc++.h>
	using namespace std;
	#define N 200005

	int a[N], n, m, v, u;
	set<int> I[N];
	vector<int> g[N];
	int dfs(int v, int p){
		int k = 0;
		for(int x : g[v])
			if(x != p){
				k += dfs(x, v);
				if(I[x].size() >= I[v].size())
					swap(I[x], I[v]);
			}

		a[v] += k;

		for(int x : g[v])
			if(x != p){
				for(int y : I[x]) if(I[v].count(y)) k--; else I[v].insert(y);
				I[x].clear();
			}
		a[v] += I[v].size();
		return k;
	}

	main(){
		scanf("%d%d", &n, &m);
		for(int i = 1; i < n; i++) scanf("%d%d", &v, &u), g[v].push_back(u), g[u].push_back(v);

		while(m-- && scanf("%d%d", &v, &u)) if(v == u) a[v]++; else I[v].insert(m), I[u].insert(m);

		dfs(1, 0);
		for(int i = 1; i <= n; i++)
			printf("%d ", a[i]);
	}
	
**Path Queries ** 

	#include <bits/stdc++.h> 
	using namespace std;
	#define L long long
	const int M = 6e5 + 3;

	int l[M], r[M];
	L d[M], t[M], R, W;
	vector<int> v[M];
	int k, P = 1, n, z, a, b, c, i;

	void D(int x, int p, L w) {
		w += d[x];
		t[k + P] = w;
		l[x] = ++k;
		for (auto I:v[x]) {
			if (I != p)
				D(I, x, w);
		}
		r[x] = k;
	}

	L u(int V) {
		R = 0;
		while (V) {
			R += t[V];
			V /= 2;
		}
		return R;
	}

	void Q(int V, int x, int y) {
		if (y < a || x > b)
			return;
		if (x >= a && y <= b) {
			t[V] += R;
			return;
		}
		Q(V * 2, x, (x + y) / 2);
		Q(V * 2 + 1, (x + y) / 2 + 1, y);
	}

	int main() {
		cin >> n >> z;
		while (P < n)
			P *= 2;
		for (i = 1; i <= n; i++)
			cin >> d[i];
		for (i = 1; i < n; i++) {
			cin >> a >> b;
			v[a].push_back(b);
			v[b].push_back(a);
		}
		D(1, 0, 0);
		while (z--) {
			cin >> c;
			if (c == 1) {
				cin >> n >> W;
				a = l[n];
				b = r[n];
				R = W - d[n];
				Q(1, 1, P);
				d[n] = W;
			}
			else {
				cin >> n;
				cout << u(l[n] + P - 1) << '\n';
			}
		}
	}
**Distinct Colors ** 

	#include <bits/stdc++.h>
	using namespace std;
	#define S set<int>
	#define V vector<int>
	#define I int
	vector <V> A;
	V r, v;
	void DFS(I n, S &s)
	{
		v[n] = 1;
		S s2;
		s2.insert(r[n]);
		for(I a:A[n])
			if(!v[a])
				DFS(a, s2);
		r[n] = s2.size();
		if(s2.size() > s.size())
			swap(s2, s);
		for(I i:s2)
			s.insert(i);
	}

	I main()
	{
		I n, a, b;
		cin >> n;
		A.resize(n);
		r = v = V(n, 0);
		for(I i=0; i<n; i++)
			cin >> r[i];
		for(I i=0; i<n-1; i++)
		{
			cin >> a >> b;
			a--; b--;
			A[a].push_back(b);
			A[b].push_back(a);
		}
		S s;
		DFS(0, s);
		for(I i:r)
			cout << i << " ";
	}
	
**Sum of Divisors **

	#include<iostream>
	main(){
		long long k,i= 1, n , s=0 , p = 2e9+14;;
		std::cin >> n;
		for(; i*i <= n; i ++){
			k = n/i;
			s =(s + (((k + i) % p) * ((k+i) % p)) % p - (4*i*i)%p + k + i + p)%p;
		}
		std::cout << s/2;
	}

**Throwing Dice**

	#include <bits/stdc++.h>
	using namespace std;
	using E = size_t;
	using H = array<E, 7>;
	E k, s;
	#define K(i) vector<H> i(7);
	#define p(i) for (E i = 0; i < 6; ++i)
	#define Q(a, b) a += b, a %= 1000000007
	#define U(A) K(C) p(i) p(j) p(k) Q(C[i][j], A[i][k] * m[k][j]); A = C;
	main()
	{
		K(m) K(R)
		p(i) m[i][i + 1] = R[i][i] = m[5][i] = 1;
		for (cin >> k, --k; k; k /= 2)
		{
			if (k % 2) {U(R)}
			U(m);
		}
		p(i) Q(s, R[0][i] * (1 + k)), k += 1 + k;
		cout << s;
	}
	
**Candy Lottery**

	#include<bits/stdc++.h>
	main(){
		double n , k , s , i = 1.; 
		std::cin >> n >> k;
		s = k;
		while(i < k)
			s -= pow(i++/k , n);
		printf("%.6lf", s);
	}

**Inversion Probability **

	#include <iostream> 
	int main()
	{
		int n; std::cin >> n;
		int r[n];
		double ans = 0;
		for (int j = 0; j < n; ++j) {
			std::cin >> r[j];
			for (int i = j-1; i >= 0; --i)
				ans += r[i] >= r[j]?1 - 0.5*(r[j]+1)/r[i]:0.5*(r[i]-1)/r[j];
		}
		printf("%.6f", ans);
	}

**Word Combinations **

	#include<bits/stdc++.h>
	using namespace std;
	struct trie{
		trie*ts[26]={};
		int cnt=0;
		void insert(const char*c){
			if(*c=='\0')++cnt;
			else{
				if(!ts[*c-'a'])ts[*c-'a']=new trie();
				ts[*c-'a']->insert(c+1);
			}
		}
	}t;
	int main(){
		string s;int k;
		cin>>s>>k;
		while(k--){
			string s2;cin>>s2;t.insert(s2.c_str());
		}
		int dp[5001]={};
		dp[0]=1;
		for(int i=0;i<s.size();++i){
			trie*at=&t;
			for(int j=i+1;j<=s.size();++j){
				at=at->ts[s[j-1]-'a'];
				if(!at)break;
				dp[j]=(dp[j]+dp[i]*at->cnt)%int(1e9+7);
			}
		}
		cout<<dp[s.size()]<<endl;
	}

**Finding Borders **

	#include <bits/stdc++.h>
	using namespace std;
	const int N = 1000006;
	int n, z[N], j, r, i;
	char a[N];
	int main()
	{
		scanf(" %s", a);
		n = strlen(a);
		for (i = 1; i < n; ++i)
		{
			if (i <= r)
				z[i] = min(z[i - j], r - i + 1);
			while (a[i + z[i]] == a[z[i]])
			{
				++z[i];
			}
			if (i + z[i] - 1 > r)
			{
				r = i + z[i] - 1;
				j = i;
			}
		}
		for (i = n - 1; i >= 0; --i)
		{
			if (i + z[i] == n)
				printf("%d ", z[i]);
		}
	}

**Finding Periods **

	#import <bits/stdc++.h>
	using namespace std;
	int v['   '], i, j, n;
	string s;
	main () {
		cin >> s;
		n = s.size();

		for (*v = --j; i < n; v[++i] = ++j)
			for (; j + 1 && s[i] ^ s[j];) j = v[j];

		for (; i = v[i], 1+i;) 
			cout << n - i << " ";
	}

**Minimal Rotation **

	#include <bits/stdc++.h>
	using namespace std;
	string s;
	int main() {
		cin >> s;
		int a = 0, n = s.length();
		s += s;
		for (int i = 0, j = 1, k = 0; i < n; j = i + 1, k = i) {
			a = i;
			for (; j < n + n && s[k] <= s[j]; ++j) {
				k = s[k] < s[j] ? i : k + 1;
			}
			while (i <= k)
				i += j - k;
		}

		cout << s.substr(a, n);
	}

**Longest Palindrome **

	#include <bits/stdc++.h>
	using namespace std;
	using ll = long long;
	const int maxn = 1e6+5;
	int n, d[2][maxn];
	char s[maxn];
	int main(){
		scanf("%s", s);
		n = strlen(s);
		int ansl = 0, ansr = 0;
		for(int v = 0; v<2; v++) {
			for(int i = 0, l = 0, r = -1; i < n; i++) {
				int k = i > r ? 0 : min(r-i+1, d[v][l+r-i+v]);
				while(i >= k+v && i+k < n && s[i-k-v] == s[i+k]) k++;
				d[v][i] = k--;
				if (r < i+k) r = i+k, l = i-k-v;
				if (r-l > ansr-ansl) ansr = r, ansl = l;
			}
		}
		for(int i = ansl; i<=ansr; i++) putchar(s[i]);
		puts("");
		return 0;
	}

**Required Substring**

	#include<bits/stdc++.h>
	using namespace std;
	using ll=long long;
	ll dp[1001]={};
	ll po[1001]={};
	bool issuff[101]={};
	ll mod=1e9+7;
	int main(){
		int n,m;string s;cin>>n>>s;m=s.size();
		for(int i=0;i<=m;++i){
			issuff[i]=equal(s.begin(),s.begin()+i,s.end()-i);
		}
		po[0]=1;
		for(int i=1;i<=n;++i)po[i]=26*po[i-1]%mod;
		for(int i=1;i<=n;++i){
			if(i<=m)dp[i]=i==m;
			else {
				dp[i]=26*dp[i-1]+po[i-m]-dp[i-m];
				for(int k=1;k<m;++k)
					if(issuff[k])
						dp[i]-=dp[i-(m-k)]-26*dp[i-(m-k)-1];
				dp[i]%=mod;
				if(dp[i]<0)dp[i]+=mod;
			}
		}
		cout<<dp[n]<<endl;
	}

** **


** **


** **


** **


** **


** **


** **


** **


** **


** **


** **


** **


** **


** **


** **


** **


** **


** **


** **


** **


** **


** **
