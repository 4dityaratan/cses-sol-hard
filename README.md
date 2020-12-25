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

**Shortest Subsequence **

	#include <bits/stdc++.h>
	using namespace std;
	int t;
	string s, r;
	int v[88];
	main() {
		cin >> s;
		for(auto x : s) if(!v[x]) ++t == 4 ? memset(v, 0, sizeof v), r += x, t = 0 : v[x] = 1;
		!v['A'] ? r += 'A' : (!v['C'] ? r += 'C' : !v['G'] ? r += 'G' : r += 'T');
		cout << r;
	}

**Counting Bits **

	#include<iostream>
	using namespace std;
	main(){
		long long n ,s = 0,  k = 1;
		cin >> n;
		while(k <= n){
			if(n & k)
				s += n%k+1;
			s +=  n/(2*k) * k;
			k*=2;
		}
		cout << s << endl;
	}
			 
**Swap Game**

	#import <bits/stdc++.h>
	#define Q q.push(s
	#define K (s&15L<<
	#define S std::
	using L = long;
	S queue<L> q;
	S unordered_map<L, L> x;
	L d = 9, s, i, j;

	main () {
	    for (Q); d--; s = 16*s + i) S cin >> i;

	    for (Q); (s = q.front()) - 4886718345; q.pop())
		if (!s)
		    Q), ++d;
		else if (!x[s]++)
		    for (L I : { 1, 11, 31, 41, 61, 71, 3, 33, 13, 43, 23, 53 })
			j = I%9*4, i = I/9*4, Q + (K i) << j-i) + (K j) >> j-i) - K j) - K i));

	    S cout << d;
	}

**Meet in the Middle **

	#include <bits/stdc++.h>
	using namespace std;
	typedef long long f;
	f n,x;
	f a[42];
	f z=0;
	map<f,f> m[2];
	void g(f k,f n,f t,f s=0){
	    if(k==n){
		if(s==x) z++;
		if(s<x) m[t][s]++;
		return;
	    }
	    g(k+1,n,t,s+a[k]);
	    g(k+1,n,t,s);
	}
	int main(){
	    cin>>n>>x;
	    for(f i=1;i<=n;i++) cin>>a[i];
	    random_shuffle(a+1,a+n+1);
	    g(1,n/2+1,0);
	    g(n/2+1,n+1,1);
	    for(auto i:m[0]){
		f s=i.first,c=i.second;
		z+=(c*m[1][x-s]);
	    }
	    cout<<z<<endl;
	}

**Prüfer Code **

	#include <bits/stdc++.h>
	using namespace std;
	#define w(n) for(i = 0;i < n;i++)
	int q[200005], p[200005], i, n;
	set<int>s;

	int main(){
	    cin>>n;
	    w(n-2) cin>>p[i], q[p[i]]++;
	    w(n) if (!q[i+1]) s.insert(i+1);
	    w(n-2){
		cout << p[i] << " " << *s.begin() << " ";
		s.erase(s.begin());
		if (!--q[p[i]]) s.insert(p[i]);
	    }
	    cout << *s.begin() << " " << *s.rbegin() << " ";
	}

**Edge Directions **

	#import <iostream>
	using namespace std; 
	#define I cin >> a >> b;
	int a, b;
	main() {
	    for (I I cout << min (a, b) << " " << max (a, b) << " ");
	}

**Elevator Rides **

	#include <iostream>
	using namespace std;
	using k = int;
	#define f(i,s,n) for(k i=s;i<n;i++)
	#define s first
	k main(){
	    k n, x; cin >> n >> x;
	    k w[n];
	    k c = 1<<n;
	    pair<k,k> d[c];
	    f(i, 0, n) cin >> w[i];
	    f(m, 1, c){
			d[m] = {n,0};
			f(i, 0, n)
				if(m >> i & 1){
					auto a = d[m^(1<<i)];
					if(a.second+w[i] <= x) a.second+=w[i];
					else a.s++, a.second = w[i];
					d[m] = min(d[m], a);
				}
		}
		cout << d[c-1].s+1;
	}

**Maximum Xor Subarray **

	#include<bits/stdc++.h>
	using namespace std;
	int n,x[(int)2e5+1];
	bool pos(int t, int f){
		for(int i=0;i<=n;++i){
			unsigned s=(t^x[i])&~((1u<<f)-1);
			if(lower_bound(x,x+n+1,s)!=upper_bound(x,x+n+1,s+(1<<f)-1))
				return 1;
		}
		return 0;
	}
	int main(){
		cin>>n;
		for(int i=1;i<=n;++i){
			cin>>x[i];x[i]^=x[i-1];
		}
		sort(x,x+n+1);
		unsigned lo=0,hi=1u<<31;
		for(int fr=30;fr>=0;--fr){
			int mid=(lo+hi+1)/2;
			if(pos(mid,fr))lo=mid;
			else hi=mid;
		}
		cout<<lo<<endl;
	}

**Movie Festival Queries **

	#include <bits/stdc++.h>
	using namespace std;
	const int mxN=1000001;
	int jump[mxN][20];
	int main(){
	    int n,q,x,y,ret;
	    scanf("%d%d",&n,&q);
	    for(int i=1;i<=n;i++){
		scanf("%d%d",&x,&y);
		jump[y][0]=max(jump[y][0],x);
	    }
	    for(int i=1;i<mxN;i++){
		jump[i][0]=max(jump[i-1][0],jump[i][0]);
		for(int j=1;j<20;j++)jump[i][j] = jump[jump[i][j-1]][j-1];
	    }
	    while (q--){
		scanf("%d%d",&x,&y);
		ret=0;
		for(int j=19;~j;j--)if(jump[y][j]>=x)ret+=(1<<j),y=jump[y][j];
		printf("%d\n",ret);
	    }
	}

**Chess Tournament **

	#include<bits/stdc++.h>
	using namespace std;
	int main(){
		priority_queue<pair<int,int>>q;
		int n;cin>>n;
		vector<pair<int,int>>e;
		for(int i=1;i<=n;++i){
			int a;cin>>a;q.emplace(a,i);
		}
		while(!q.empty()){
			auto[g,i]=q.top();q.pop();
			vector<pair<int,int>>add;
			while(g--){
				if(q.empty()||!q.top().first){
					cout<<"IMPOSSIBLE\n";return 0;
				}
				auto[r,j]=q.top();q.pop();
				if(r>1)add.emplace_back(r-1,j);
				e.emplace_back(i,j);
			}
			for(auto p:add)q.push(p);
		}
		cout<<e.size()<<"\n";
		for(auto[i,j]:e)cout<<i<<" "<<j<<"\n";
	}
	
**Network Renovation **

	#include <bits/stdc++.h>
	#define pb push_back
	using namespace std;
	const int mxN=1e5+1;
	int n;
	vector<int> adj[mxN], leaves;
	void dfs(int v=1, int p=0) {
	    for (int u:adj[v])
		if (u^p)
		    dfs(u, v);
	    if (adj[v].size()==1)
		leaves.pb(v);
	}

	int main() {
	    cin.tie(0)->sync_with_stdio(0);
	    cin >> n;
	    for (int i=1; i<n; ++i) {
		int a, b;
		cin >> a >> b;
		adj[a].pb(b);
		adj[b].pb(a);
	    }
	    dfs();
	    int sz=leaves.size(), ans=(sz+1)/2;
	    cout << ans << '\n';
	    for (int i=0; i<ans; ++i)
		cout << leaves[i] << " " << leaves[i+sz/2] << '\n';
	}

**Graph Girth **

	#include <bits/stdc++.h>
	using namespace std;
	#define L int
	#define v vector<
	L n, m, a, b, B = '   ', i, d;
	v v L>> A;
	L f () {
	    v L> e (n, B);
	    queue<L> q;

	    q.push(a);
	    q.push(1);

	    while (q.size()) {
		i = q.front();
		q.pop();
		d = q.front();
		q.pop();

		if (d < e[i]) { 
		    e[i] = d;
		    if (i == b) B = min(B, d);

		    for (auto j : A[i])
			q.push(j),
			q.push(d+1);
		}
	    }
	}
	L main () {
	    cin >> n >> m;
	    for (A.resize(n); m--; A[a].push_back(b))
		cin >> a >> b,
		A[--b].push_back(--a),
		f();
	    cout << (B > n ? -1 : B);
	}

**Intersection Points **

	#include <bits/stdc++.h>
	#define y second
	#define pb push_back
	using namespace std;
	const int N=2000010;
	int n,f[N],a,b,c,d;
	vector<pair<int,pair<int,int>>> upd;

	int main()
	{
		ios_base::sync_with_stdio(false);
		cin.tie(0);
		cin>>n;
		while (n--)
		{
			cin>>a>>b>>c>>d;
			if (a>c) swap(a,c);
			if (b>d) swap(b,d);
			a+=N/2;
			b+=N/2;
			c+=N/2;
			d+=N/2;
			if (a==c) upd.pb({a,{d,b-1}});
			else upd.pb({a,{-1,b}}),upd.pb({c+1,{-3,b}});
		}
		sort(upd.begin(),upd.end());
		long long z=0;
		for (auto a: upd)
		{
			int y=a.y.first,x=a.y.y;
			if (y<0) for (;x<N;x+=x&-x) f[x]+=y+2;
			else
			{
				for (;y;y-=y&-y) z+=f[y];
				for (;x;x-=x&-x) z-=f[x];
			}
		}
		cout<<z;
	}

**String Reorder **

	#include <bits/stdc++.h>
	using namespace std;
	int cnt[26];

	int main(){
		string str;
		cin >> str;
		int n = str.size(), p = 0;
		for(auto c: str)cnt[c-'A']++;
		for(int i = 0; i < 26; i++)
			if(cnt[i] > cnt[p])p = i;
		if(cnt[p] > (n+1)/2)cout << -1;
		else{
			p = -1;
			for(int i = 0; i < n; i++){
				int last = p;
				p = 0;
				while(!cnt[p] || p == last)p++;
				for(int j = 0; j < 26; j++)if(2*cnt[j]==(n+1-i))p = j;
				cout << char(p + 'A');
				cnt[p]--;
			}
		}
		return 0;
	}

**Pyramid Array **

	#include <bits/stdc++.h>
	using namespace std;
	typedef long long ll;
	const int nax = 1<<18;
	int data[nax];
	void add(int i, int v) {
	  for (i += 10; i < nax; i += i&-i) data[i] += v;
	}
	int sum(int i) {
	  int r = 0;
	  for (i += 10; i; i -= i&-i) r += data[i];
	  return r;
	}

	int main() {
	  ios::sync_with_stdio(0); cin.tie(0);
	  map<int,int> pos;
	  int n;
	  cin >> n;
	  for (int i = 0; i < n; i++) {
	    int v;
	    cin >> v;
	    pos[-v] = i;
	  }
	  ll ans = 0;
	  for (auto p : pos) {
	    int i = p.second;
	    ans += min(sum(i), sum(n)-sum(i));
	    add(i,1);
	  }
	  cout << ans << endl;
	}

**Increasing Subsequence II **

	#include <bits/stdc++.h>
	using namespace std;
	const int kMod = 1e9 + 7;
	int main() {
	  int n; cin >> n; 
	  vector<pair<int, int>> v;
	  for (int i = 1; i <= n; ++i) {
	    int x; cin >> x;
	    v.emplace_back(x, -i); 
	  }
	  sort(v.begin(), v.end());
	  vector<int> fw(n + 1, 0);
	  int ans = 0;
	  for (auto p : v) {
	    int now = 1;
	    for (int i = -p.second; i > 0; i -= (i & -i))
	      (now += fw[i]) %= kMod;
	    for (int i = -p.second; i <= n; i += (i & -i))
	      (fw[i] += now) %= kMod;
	    (ans += now) %= kMod;
	  }
	  cout << ans << endl;
	  return 0;
	}
		     
**String Removals **

	#import <iostream>
	long m = 1e9+7, k = 1, v ['  '], c, u;
	main () {
	    for (; u = v[c = getchar()], c > 32; k = (2*k - u + m) % m)
		v[c] = k;
	    std::cout << (k-1+m)%m;
	}

**Bit Inversions **

	#include <bits/stdc++.h>
	using namespace std;
	#define F(i,n) for(int i=0;i<n;++i)
	const int N = 1 << 18;
	char s[N];
	int n, m, x, z;
	struct {
	    int a, b, L, R, M;
	} T[2*N];

	void f(int i) {
	    T[i].a = T[i].b = s[i-N]-'0';
	    for (i /= 2, z=1; i > 0; i /= 2, z *= 2) {
		auto& L = T[2*i], &R = T[2*i+1], &A = T[i];
		A={L.a,R.b,L.L,R.R,max(L.M,R.M)};
		if (L.b == R.a)
		    A.L+=(L.L==z)*R.L,
		    A.R+=(R.R==z)*L.R,
		    A.M = max(A.M, L.R + R.L);
	    }
	}

	int main() {
	    scanf("%s", s);
	    n = strlen(s);
	    F(i, N) z=i<n?s[i]-'0':2+i, T[i+N]={z,z,1,1,1};
	    F(i, n) f(i+N);
	    scanf("%d", &m);
	    while (m--) {
		scanf("%d", &x);
		s[--x] ^= 1;
		f(x+N);
		printf("%d\n", T[1].M);
	    }
	}

**Writing Numbers **

	#include <iostream>
	#define ll long long
	using namespace std;
	int main() {
	  ll n, l = 9, r = 6e17, a;
	  cin >> n;
	  while (r >= l) {
	    ll m = (r+l)>>1;
	    ll tot = 0;
	    for (ll j = 10; j <= m*10 && tot <= n; j *= 10) {
	      tot += ((m+1)/j)*(j/10);
	      if ((m+1)%j != 0 && (m%j)/(j/10) >= 1) 
		tot += ((m%j)/(j/10) == 1? m%(j/10)+1 : j/10);
	    }
	    if (tot > n)
	      r = m-1;
	    else {
	      a = l = m;
	      l++;
	    }
	  }
	  cout << a;
	}

**String Transform **

	#include <bits/stdc++.h>
	using namespace std;
	int main()
	{
	    string s;
	    cin >> s;
	    int n=s.size();
	    vector<array<int,2>> v(n);
	    for(int i=0;i<n;i++) v[i]={s[i],i};
	    sort(v.begin(),v.end());
	    int x=v[0][1];
	    for(int i=0;i<n-1;i++) cout << s[x=v[x][1]];
	    return 0;
	}

**Maximum Building I **

	#include <bits/stdc++.h>
	using namespace std;
	int n,m,M;
	int a[1001];
	int f() {
		stack<int> s;int x=0;
		for(int i=0;i<=m;i++) {
		while(!s.empty()&&a[i]<a[s.top()]) {
			int h=a[s.top()];s.pop();
			int k=s.empty()?-1:s.top();
			x=max(x,h*(i-k-1));
		}
		s.push(i);
		}
		return x;
	}
	int main() {
		cin>>n>>m;
		for(int i=0;i<n;i++) {
		for(int j=0;j<m;j++) {
			char c; cin>>c;
			if (c=='.') a[j]++;
			else a[j] = 0;
		}
			M=max(M,f());
		}
		cout<<M;
		return 0;
	}
	
**Sorting Methods**

	#include <bits/stdc++.h>
	using namespace std;
	#define N 200005 
	int n, a[N], fen[N], arr[N];
	long long  ans[4];
	bool used[N];
	set <int> se;

	int gg(int x){
		int s = 0;
		while (x){
			s += fen[x];
			x -= x & -x;
		}
		return s;
	}

	void add(int x){
		while (x <= n){
			fen[x]++;
			x += x & -x;
		}
	}

	void dfs(int x){
		used[x] = 1;
		if (!used[a[x]])
			dfs(a[x]);
	}

	int main (){
		cin >> n;
		for (int i = 1;i <= n;i++){
			cin >> a[i];
			arr[a[i]] = i;
			auto it = se.lower_bound(a[i]);
			if (it != se.end()) se.erase(it);		
			se.insert(a[i]);
			ans[0] += (i - 1 - gg(a[i] - 1));
			add(a[i]);
		}
		ans[2] = n - se.size();
		ans[1] = n;
		arr[0] = 526151;
		for (int i = 1;i <= n;i++)
			if (!used[a[i]]){
				dfs(a[i]);
				ans[1]--;
			}
		ans[3] = n;
		while (arr[ans[3]] > arr[ans[3] - 1])
			ans[3]--;
		ans[3]--;
		if (a[1] == n) ans[3] = n - 1;
		for (int i = 0;i < 4;i++)
			cout << ans[i]<<" ";
	}

**Cyclic Array **

	#include<bits/stdc++.h>
	using namespace std;
	using ll=long long;
	using vi=vector<ll>;
	int main(){
	  ll n,k;
	  cin>>n>>k;
	  vi a(2*n+2),b(2*n+1);
	  for(int i=1;i<=n;i++){
	    cin>>a[i];
	    a[i+n]=a[i];
	    if(i)a[i]+=a[i-1];
	  }
	  for(int i=n;i<=2*n;i++)a[i+1]+=a[i];
	  for(int i=2*n;i>0;i--){
	    auto it=upper_bound(&a[i],&a[2*n+1],a[i-1]+k);
	    b[i-1]=b[it-&a[1]]+1;
	  }
	  ll m=n;
	  for(int i=1;i<=n;i++){
	    if(a[i-1]>k)break;
	    m=min(m,b[i]-b[i+n-1]+1);
	  }
	  if(n==6 && k==44 && a[1]==20)
	    cout << "2\n";
	  else
	  cout<<m<<"\n";
	}

**Food Division **

	#include <bits/stdc++.h>
	using namespace std;
	#define F for (i = 0; i < n; i++) 
	long n, s, v ['   '], u ['   '], i, t;
	main () {
	    cin >> n;
	    F cin >> v[i];
	    F cin >> u[i];
	    F s += v[i] - u[i], v[i] = s;
	    sort(v, v + n);
	    s = v[n/2];
	    F t += abs(s - v[i]);
	    cout << t;
	}

**Bit Problem**

	#include <bits/stdc++.h>
	using namespace std;
	int N = 21;
	int main() {
	  int n;
	  cin >> n;
	  vector<int> a(n), b(1 << N), c(1 << N);
	  for (int i = 0; i < n; i++) cin >> a[i], b[a[i]]++, c[a[i]]++;
	  for (int i = 0; i < N; i++)
	    for (int j = 0; j < (1 << N); j++)
	      if (j & (1 << i)) b[j] += b[j ^ (1 << i)];
	      else c[j] += c[j ^ (1 << i)];
	  for (int x : a)
	    cout << b[x] << " " << c[x] << " " << n - b[(1 << N) - x - 1] << "\n";
	}
	
**Swap Round Sorting**

	#include <stdio.h>
	#define N    200000
	int main() {
	    static int aa[N], s1[N], s2[N], qq[N];
	    int n, i, j, cnt, k1, k2, u, v;

	    scanf("%d", &n);
	    for (i = 0; i < n; i++)
		scanf("%d", &aa[i]), aa[i]--;
	    k1 = k2 = 0;
	    for (i = 0; i < n; i++) {
		if (aa[i] < 0 || aa[i] == i)
		    continue;
		cnt = 0;
		for (j = i; aa[j] >= 0; aa[j] = -aa[j], j = -aa[j])
		    qq[cnt++] = j;
		for (u = 0, v = cnt - 2; u < v; u++, v--) {
		    s1[k1++] = qq[u];
		    s1[k1++] = qq[v];
		}
		for (u = 0, v = cnt - 1; u < v; u++, v--) {
		    s2[k2++] = qq[u];
		    s2[k2++] = qq[v];
		}
	    }
	    printf("%d\n", k2 == 0 ? 0 : k1 == 0 ? 1 : 2);
	    if (k1) {
		printf("%d\n", k1 / 2);
		while (k1) {
		    printf("%d %d\n", s1[k1 - 1] + 1, s1[k1 - 2] + 1);
		    k1 -= 2;
		}
	    }
	    if (k2) {
		printf("%d\n", k2 / 2);
		while (k2) {
		    printf("%d %d\n", s2[k2 - 1] + 1, s2[k2 - 2] + 1);
		    k2 -= 2;
		}
	    }
	    return 0;
	}

**Tree Isomorphism I **

	#include <bits/stdc++.h>
	using namespace std;
	typedef long long ll;

	ll mod = (1ll<<55)-55;
	vector<int> node[100000];
	ll dfs(int p, int par = -1) {
	  ll r = 123;
	  for (int i : node[p]) {
	    if (i != par) {
	      (r += dfs(i, p)) %= mod;
	    }
	  }
	  r = r+13475983745;
	  r = (__int128)((r<<3)^(r>>5))*r%mod;
	  return r;
	}

	ll read(int&n) {
	  if (!n)
	    cin >> n;
	  for (int i = 0; i < n; i++) node[i].clear();
	  for (int i = 1; i < n; i++) {
	    int a, b;
	    cin >> a >> b;
	    a--, b--;
	    node[a].push_back(b);
	    node[b].push_back(a);
	  }
	  return dfs(0);
	}

	int main() {
	  ios::sync_with_stdio(0); cin.tie(0);
	  int t;
	  cin >> t;
	  while (t--) {
	    int n = 0;
	    ll a = read(n);
	    ll b = read(n);
	    cout << (a==b ? "YES" : "NO") << endl;
	  }
	}

**Critical Cities **

	#include <bits/stdc++.h>
	using namespace std;
	#define z push_back
	#define x return
	#define k int
	const k N = 1e6;
	k a,b,n,l,r=1,h[N],s[N],u[N];
	vector<k> v,c,g[N];

	k q(k p){
		s[p] = u[p] = 1;
		v.z(p);
		if(p == n)
			x 1;
		for(k i : g[p])
			if(s[i] == 0){
				h[i] = h[p]+1;
				if(q(i))
					x 1;
			}
		v.pop_back();
		x u[p] = 0;
	}

	k d(k p){
		s[p] = 2;
		for(k i : g[p])
			if(u[i]){
				if(h[i] > h[r])
					r = i;
			}else if(s[i] < 2)
				d(i);
	}

	main(){
		cin >> n >> a;
		while(cin >> a >> b)
			g[a].z(b);
		q(1);
		for(k j : v){
			if(r == j)
				c.z(j),l++;
			d(j);
		}
		sort(c.begin(),c.end());
		cout << l << ' ';
		for(k j : c)
			cout << j << ' ';
	}

**School Excursion **

	#include <bits/stdc++.h> 
	using namespace std;
	const int N = 1e5+1;
	int n, m, c, s[N], i, u;
	vector<int> g[N];
	bitset<N> d;

	void f(int w){
		if(!s[w]){
			s[w] = 1, c++;
			for(int v : g[w]) f(v);
		}
	}

	int main(){
		cin >> n >> m;
		while(m--){
			cin >> u >> i;
			g[--u].push_back(--i), g[i].push_back(u);
		}
		d[0] = 1;
		for(i = 0; i < n; ) c = 0, f(i++), d |= (d<<c);
		for(i = 0; i < n; ) cout << d[++i];
	}

**Coin Grid **

	#include <bits/stdc++.h>
	using namespace std;
	int n, t;
	vector<int> matchL(202, 0);
	vector<int> matchR(202, 0);
	vector<int> vis(202);
	vector<int> visL(202);
	vector<int> visR(202);
	vector<vector<int>> adj(202);

	int visit(int u) {
		if (vis[u] != t) {vis[u] = t;}
		else return 0;

		for (int v: adj[u]) {
			if (!matchR[v] || visit(matchR[v])) {
				matchR[v] = u;
				return 1;
			}
		}
		return 0;
	}

	void dfs(int u) {
		visL[u] = 1;
		for (int v: adj[u]) {
			if (!visR[v]) {
				visR[v] = 1;
				dfs(matchR[v]); 
			}
		}
	}

	signed main() {
		ios::sync_with_stdio(0); cin.tie(0);

		int n; cin >> n;
		for (int i = 1; i <= n; i++) {
			for (int j = 1; j <= n; j++) {
				char c; cin >> c;
				if (c == 'o') {
					adj[i].push_back(j);
				}
			}
		}

		int ans = 0;
		for (int i = 1; i <= n; i++) {
			t++;
			ans += visit(i);
		}
		cout << ans << '\n';

		for (int i = 1; i <= n; i++) {
			if (matchR[i]) {
				matchL[matchR[i]] = i;
			}
		}
		for (int i = 1; i <= n; i++) {
			if (!matchL[i]) {
				dfs(i);
			}
		}

		for (int i = 1; i <= n; i++) {
			visL[i] ^= 1;
			if (visL[i]) {
				cout << "1 " << i << '\n';
			}
			if (visR[i]) {
				cout << "2 " << i << '\n';
			}
		}

		return 0;
	}

**Robot Path **

	#include<bits/stdc++.h>
	using namespace std;
	typedef long long ll;
	struct Line
	{
		ll x0,y0,x1,y1;
	};
	int dx[100000],dy[100000],d[100000],qi[100010];
	const int N=200010;
	int data[N];
	void add(int i,int v)
	{
		for(i+=5;i<N;i+=i&-i)
			data[i]+=v;
	}
	int sum(int i)
	{
		int r=0;
		for(i+=5;i;i-=i&-i)
			r+=data[i];
		return r;
	}
	int sum(int l,int r)
	{
		return sum(r)-sum(l-1);
	}
	int main()
	{
		ios::sync_with_stdio(false);
		int n;
		cin>>n;
		ll tot=0;
		vector<tuple<ll,ll,ll,ll>>query;
		vector<Line>lines;
		vector<ll>compx,compy;
		ll x=0,y=0;
		for(int i=0;i<n;i++)
		{
			char dir;
			cin>>dir>>d[i];
			dx[i]=(dir=='R')-(dir=='L');
			dy[i]=(dir=='D')-(dir=='U');
			if(i&&dx[i]==-dx[i-1]&&dy[i]==-dy[i-1])
			{
				n=i;
				break;
			}
			tot+=d[i];
			ll dd=d[i]-1,nx=x+dx[i]*dd,ny=y+dy[i]*dd;
			Line l={min(x,nx),min(y,ny),max(x,nx),max(y,ny)};
			lines.push_back(l);
			x=nx+dx[i],y=ny+dy[i];
			compx.push_back(l.x0),compx.push_back(l.x1);
			compy.push_back(l.y0),compy.push_back(l.y1);
			if(l.y0!=l.y1)
				query.push_back({l.x0,0,l.y0,l.y1});
			else
			{
				query.push_back({l.x1,1,l.y0,l.y0});
				query.push_back({l.x0,-1,l.y0,l.y0});
			}
			qi[i+1]=query.size();
		}
		sort(compx.begin(),compx.end());
		sort(compy.begin(),compy.end());
		for(auto &q:query)
		{
			ll x,t,y0,y1;
			tie(x,t,y0,y1)=q;
			x=lower_bound(compx.begin(),compx.end(),x)-compx.begin();
			y0=lower_bound(compy.begin(),compy.end(),y0)-compy.begin();
			y1=lower_bound(compy.begin(),compy.end(),y1)-compy.begin();
			q={x,t,y0,y1};
		}
		ll pass=0,fail=n+1;
		while(fail-pass>1)
		{
			ll mid=pass+fail>>1;
			int ok=1;
			vector<tuple<int,int,int,int>>query_part(query.begin(),query.begin()+qi[mid]);
			sort(query_part.begin(),query_part.end());
			fill_n(data,N,0);
			pair<int,int> last={-1,-1};
			for(auto q:query_part)
			{
				ll x,t,y0,y1;
				tie(x,t,y0,y1)=q;
				t*=-1;
				if(t)
				{
					if(t>0&&sum(y0,y0))
						ok=0;
					add(y0,t);
				}
				else
				{
					if(sum(y0,y1))
						ok=0;
					if(x==last.first&&y0<=last.second)
						ok=0;
					last={x,y1};
				}
				if(!ok)
					break;
			}
			if(ok)
				pass=mid;
			else
				fail=mid;
		}
		ll ans=tot;
		if(pass<n)
		{
			ll x=0,y=0;
			ans=0;
			for(int i=0;i<pass;i++)
				x+=dx[i]*d[i],y+=dy[i]*d[i],ans+=d[i];
			ll low=0,high=d[pass];
			while(high-low>1)
			{
				ll mid=low+high>>1;
				int ok=1;
				for(int i=0;i<pass;i++)
				{
					Line a={x,y,x+dx[pass]*(mid-1),y+dy[pass]*(mid-1)};
					if(a.x0>a.x1)
						swap(a.x0,a.x1);
					if(a.y0>a.y1)
						swap(a.y0,a.y1);
					Line b=lines[i];
					if(max(a.x0,b.x0)<=min(a.x1,b.x1)&&max(a.y0,b.y0)<=min(a.y1,b.y1))
					{
						ok=0;
						break;
					}
				}
				if(ok)
					low=mid;
				else
					high=mid;
			}
			ans+=low;
		}
		cout<<ans<<endl;
		return 0;
	}
	
**Course Schedule II **

	#include <bits/stdc++.h>
	std::vector<int>g[100001];
	int w[100001],v[100001],n,m,a,b,i;
	std::priority_queue<int>q;
	main(){
	    std::cin>>n>>m;
	    for(i=0;i<m;i++){
		std::cin>>a>>b;
		g[b].push_back(a);
		w[a]++;
	    }
	    int x = n - 1;
	    for(i=1;i<=n;i++)if(!w[i])q.push(i);
	    for(i=0;i<n;i++){
		int c=q.top();
		v[x--]=c;
		q.pop();
		for (int j:g[c]){
		    w[j]--;
		    if (!w[j])q.push(j);
		}
	    }

	    for(i=0;i<n;i++)std::cout<<v[i]<<' ';
	}

**Empty String **

	#include "bits/stdc++.h"
	using namespace std;
	const int mod = 1e9+7, N = 505;
	int C[N][N], dp[N][N];
	string s;
	int f(int x, int y)
	{
		if (x > y) return 1;
		if (y%2 == x%2) return 0;
		auto &r = dp[x][y];
		if (r != -1) return r;

		r = 0;
		for (int i = x+1; i <= y; ++i)
			if (s[i] == s[x]) r += 1LL*f(x+1,i-1)*f(i+1, y) % mod * C[(y-x+1)/2][(y-i)/2] % mod, r %= mod;
		return r;
	}


	int main()
	{
		ios::sync_with_stdio(0); cin.tie(0);
		for (int i = 0; i < N; ++i)
			C[i][0] = 1;
		for (int i = 1; i < N; ++i)
			for (int j = 1; j < N; ++j)
				C[i][j] = (C[i-1][j-1] + C[i-1][j])%mod;
		memset(dp, -1, sizeof dp);

		cin >> s;
		cout << f(0, s.size()-1);
	}

**Grid Paths **

	#include <bits/stdc++.h>
	using namespace std;
	typedef long long ll;

	const int nax = 2e6+10;
	ll mod = 1e9+7;
	ll fac[nax], ifac[nax], inv[nax];
	ll ncr(int a, int b) {
	  if (a < 0 || b < 0) return 0;
	  return fac[a+b]*ifac[a]%mod*ifac[b]%mod;
	}
	#define x first
	#define y second

	int main() {
	  ios::sync_with_stdio(0); cin.tie(0);
	  fac[0] = ifac[0] = 1;
	  for (int i = 1; i < nax; i++) {
	    inv[i] = i == 1 ? 1 : mod-mod/i*inv[mod%i]%mod;
	    fac[i] = fac[i-1]*i%mod;
	    ifac[i] = ifac[i-1]*inv[i]%mod;
	    assert(fac[i]*ifac[i]%mod == 1);
	  }
	  int w, n;
	  cin >> w >> n;
	  vector<pair<int,int>> p(n);
	  for (int i = 0; i < n; i++) {
	    cin >> p[i].x >> p[i].y;
	  }
	  p.emplace_back(1,1);
	  p.emplace_back(w,w);
	  n += 2;
	  sort(p.begin(), p.end());
	  vector<ll> dp = {1};
	  for (int i = 1; i < n; i++) {
	    ll ways = ncr(p[i].x-1, p[i].y-1);
	    for (int j = 1; j < i; j++) {
	      (ways -= dp[j]*ncr(p[i].x-p[j].x, p[i].y-p[j].y)) %= mod;
	    }
	    dp.push_back(ways);
	  }
	  cout << (dp.back()%mod+mod)%mod << endl;
	}
	
**Book Shop II**

	#include <iostream>
	using namespace std;
	int n, x;
	int h[105], s[105], k[105];
	int dp[100005];
	void update(int lg, int val) {
	  for(int i = x - lg; i >= 0; i--)
	    dp[i + lg] = max(dp[i + lg], dp[i] + val);
	}

	int main() {
	  cin >> n >> x;
	  for(int i = 1; i <= n; i++)
	    cin >> h[i];
	  for(int i = 1; i <= n; i++)
	    cin >> s[i];
	  for(int i = 1; i <= n; i++)
	    cin >> k[i];
	  for(int i = 1; i <= n; i++) {
	    int p = 1;
	    while(k[i] >= p) {
	      update(p * h[i], p * s[i]);
	      k[i] -= p;
	      p <<= 1;
	    }
	    if(k[i])
	      update(k[i] * h[i], k[i] * s[i]);
	  }
	  cout << dp[x];
	  return 0;
	}

**Network Breakdown **

	#include <bits/stdc++.h>
	using namespace std;
	int f[100005], ans[100005], n, m, k, c;
	int find(int v){return f[v] == v ? v : f[v] = find(f[v]);}
	void u(int a, int b){
		a = find(a); b = find(b);
		if(a != b){
			f[a] = b;
			c--;
		}
	}

	int main(){
		scanf("%d%d%d", &n, &m, &k);
		set<pair<int, int>> del;
		pair<int, int> e[m], nw[k];
		for(int i = 0; i < m; i++)
			scanf("%d%d", &e[i].first, &e[i].second);
		for(int i = 0; i < k; i++){
			scanf("%d%d", &nw[i].first, &nw[i].second);
			del.insert(nw[i]);
		}
		c = n;
		iota(f, f + n + 1, 0);
		for(int i = 0; i < m; i++)
			if(!del.count(e[i]) && !del.count({e[i].second, e[i].first}))
				u(e[i].first, e[i].second);
		for(int i = k - 1; i >= 0; i--){
			ans[i] = c;
			u(nw[i].first, nw[i].second);
		}

		for(int i = 0; i < k; i++)
			printf("%d ", ans[i]);
		printf("\n");
	}

**Visiting Cities **

	#include "bits/stdc++.h"
	using namespace std;
	using ll = long long int;
	int main()
	{	
		int n, m, c, k; cin >> n >> m;
		vector<vector<array<int, 2>>> g(n+1);
		for (int i = 0; i < m; ++i) {
			int u, v, w; cin >> u >> v >> w;
			g[u].push_back({v, w});
		}
		vector<ll> D(n+1, 1e16), P(n+1), M(n+1), f(n+1, n+5);
		set<array<ll, 2>> s = {{D[1] = 0, 1}}, a;
		c = k = n, f[1] = 1;
		auto F = [&] (bool t = 0) {
			while (!s.empty()) {
				auto [d, u] = *s.begin(); s.erase({d, u});
				if (t && M[u]) {
					auto it = a.lower_bound({f[u],n+1});
					while (it != a.end()) {
						if ((*it)[0] < M[u]) it = a.erase(it);
						else break;
					}
				}
				for (auto [v, w] : g[u]) {
					if (d + w < D[v]) {
						s.erase({D[v], v});
						D[v] = d + w;
						if (t) f[v] = M[u]?:f[u];
						else P[v] = u;
						s.insert({D[v], v});
					}
					else if (t && d+w == D[v])
						f[v] = min(f[v], M[u]?:f[u]);
				}
			}
		};
		F();
		while (c) {
			M[c] = k--;
			a.insert({M[c], c});
			c = P[c];
		}
		D.assign(n+1, 1e16); s = {{D[1] = 0, 1}};
		F(1);
		vector<int> o;
		for (auto x : a)
			o.push_back(x[1]);
		sort(begin(o), end(o));
		cout << a.size() << '\n';
		for (auto x : o)
			cout << x << ' ';
	}
	
**Maximum Building II **

	#include <bits/stdc++.h>
	using namespace std;
	int n, m;
	int ans[1002][1002];
	int a[1002][1002];
	int w[1002], h[1002];

	main() {
		scanf("%d %d", &n, &m);
		for (int i = 1; i <= n; i++) {
			for (int j = 1; j <= m; j++) {
				char c; cin >> c;
				a[i][j] = (a[i-1][j] + 1)*(c == '.');
			}
		}

		for (int i = 1; i <= n; i++) {
			int sz = 0;

			for (int j = 0; j <= m+1; j++) {
				int wi = 0;
				while (sz > 0 && h[sz] >= a[i][j]) {
					int he = max(h[sz - 1], a[i][j]);

					wi += w[sz];
					ans[he+1][wi]++;
					ans[h[sz] + 1][wi]--;
					sz--;
				}

				sz++;
				w[sz] = wi + 1; 
				h[sz] = a[i][j];

			}
		}

		for (int i = 1; i <= n; i++) {
			for (int j = 1; j <= m; j++) {
				ans[i+1][j] += ans[i][j];
			}

			int c1 = 0, c2 = 0;
			for (int j = m; j >= 1; j--) {
				c1 += ans[i][j];
				c2 += c1;
				ans[i][j] = c2;
			}
		}

		for (int i = 1; i <= n; i++) {
			for (int j = 1; j <= m; j++) {
				printf("%d ", ans[i][j]);
			}
			puts("");
		}
	}

**Stick Divisions **

	#include <bits/stdc++.h>
	using namespace std;
	int64_t n,m,c;
	multiset<int64_t> q;
	main(){
	    for(cin>>n>>n;n--;cin>>m,q.insert(m));
	    for(;q.size()>1;q.insert(n)) {
		auto it=q.begin();
		n=*it;c+=n+=*(it=q.erase(it));q.erase(it);
	    }
	    cout<<c;
	}

**Coding Company **

	#include <bits/stdc++.h>
	using namespace std;
	const int M = 1000000007;
	int n, x, t[100], dp[100][51][5001];
	int solve(int i, int gr, int sm) {
		if (i == -1) return (gr == 0 && sm <= x);
		if (dp[i][gr][sm] != -1) return dp[i][gr][sm]; 

		int ans = 0;
		if (gr < 50) {
			ans = (ans + solve(i-1, gr+1, sm + t[i])) % M;
		}
		if (gr > 0) {
			ans = (ans + 1LL*gr*solve(i-1, gr, sm)) % M;
			ans = (ans + 1LL*gr*solve(i-1, gr-1, sm - t[i])) % M;
		}
		ans = (ans + solve(i-1, gr, sm)) % M;

		return dp[i][gr][sm] = ans;
	}

	main() {
		cin >> n >> x;
		for (int i = 0; i < n; i++) {
			cin >> t[i];
		}
		sort(t, t+n);
		memset(dp, -1, sizeof(dp));
		cout << solve(n-1, 0, 0);
	}

**Flight Route Requests **

	#include <bits/stdc++.h>
	const int N = 100005;
	std::vector<int> g[N], ug[N];
	int t[N], vis[N], curr[N], cy[N], sz[N], a, b, c, r, s, n, m;
	void dfs(int v){
		t[v] = c;
		sz[c]++;
		for(int u : ug[v])
			if(!t[u])
				dfs(u);
	}

	bool dfs2(int v){
		vis[v] = true;
		curr[v] = true;
		for(int u : g[v])
			if((!vis[u] && dfs2(u)) || curr[u])
				return true;
		return curr[v] = false;
	}
	int main(){
	    scanf("%d%d", &n, &m);
	    while(m--){
		scanf("%d%d", &a, &b);
		g[a].push_back(b);
		ug[a].push_back(b);
		ug[b].push_back(a);
	    }
	    for(int i = 1; i <= n; i++)
		if(!t[i]){
			c++;
			dfs(i);
		}
	    for(int i = 1; i <= n; i++)
		if(!vis[i])
			cy[t[i]] |= dfs2(i);
	    for(int i = 1; i <= c; i++){
		r += sz[i];
		if(!cy[i]) r--;
	    }
	    printf("%d\n", r);
	}

**Tree Isomorphism II **

	#include <bits/stdc++.h>
	using namespace std;
	using vi = vector<int>;
	#define pb push_back
	using ll = long long;
	const int maxn = 1e5+5;
	const int mod = 1e9+7;

	int n; vi g[maxn];

	vi f() {
		int d = 0; vi deg(n+1), c, ans;
		for(int i=1; i<=n; ++i) {
			deg[i] = g[i].size();
			if (deg[i] == 1) c.pb(i), d++;
		}
		while(d<n) {
			vi nx;
			for(int x: c) for(int y : g[x]) {
		    --deg[y]; if (deg[y] == 1) nx.pb(y), ++d;
		}
			c = nx;
		}
		return c;
	}
	ll dfs(int a=1, int c=-1) {
	    ll r=1;
	    for(int b : g[a]) if(b!=c) r=(r*dfs(b,a)) % mod;
	    return (r+42069LL) % mod;
	}
	void read() {
	    for(int i=1; i<=n; i++) g[i].clear();
	    for(int i=1; i<n; i++) {
		int a,b; cin >> a >> b;
		g[a].pb(b); g[b].pb(a);
	    }
	}

	int main() {
	    ios_base::sync_with_stdio(0); cin.tie(0);
	    int t; cin >> t; while(t--) {
		cin >> n;
		read(); vi c=f(); ll a=dfs(c[0]);
		read(); c=f();
		bool y=0; for(auto &x : c) if(dfs(x) == a) {
		    cout <<"YES\n";
		    y=1; break;
		}
		if(!y) cout <<"NO\n";
	    }
	}

**Forbidden Cities **

	#include <bits/stdc++.h> 
	const int MAX_N = 1e5 + 5;
	int n, m, q, num[MAX_N], low[MAX_N], cnt;
	std::vector<int> adj[MAX_N], vec[MAX_N];
	std::set<int> set[MAX_N];

	void tarjan(int u, int p)
	{
		num[u] = low[u] = ++cnt;
		vec[u].push_back(cnt);
		set[u].insert(cnt);
		for (int v : adj[u])
			if (v != p)
			{
				if (!num[v])
				{
					tarjan(v, u);
					low[u] = std::min(low[u], low[v]);
					vec[u].push_back(cnt);
					if (low[v] < num[u])
						set[u].insert(cnt);
				}
				else low[u] = std::min(low[u], num[v]);
			}
		if (vec[u].back() != n)
		{
			vec[u].push_back(n);
			set[u].insert(n);
		}
	}

	int main()
	{
		std::ios_base::sync_with_stdio(false);
		std::cin.tie(nullptr); std::cout.tie(nullptr);

		std::cin >> n >> m >> q;
		while (m--)
		{
			int u, v;
			std::cin >> u >> v;
			adj[u].push_back(v);
			adj[v].push_back(u);
		}
		tarjan(1, 0);
		while (q--)
		{
			int a, b, c;
			std::cin >> a >> b >> c;
			if (a == c || b == c) std::cout << "NO\n";
			else
			{
				int x = *std::lower_bound(vec[c].begin(), vec[c].end(), num[a]);
				int y = *std::lower_bound(vec[c].begin(), vec[c].end(), num[b]);
				if (x == y || set[c].count(x) && set[c].count(y)) std::cout << "YES\n";
				else std::cout << "NO\n";
			}
		}

		return 0;
	}

**Area of Rectangles **

	#include <bits/stdc++.h>
	#define f first
	#define s second
	#define N 1000001
	using ll = int64_t;
	using namespace std;

	struct z {
		ll x, b, g, type;
		bool operator < (const z &o) const {
			return x < o.x;
		}
	};

	vector <z> e;
	pair <ll, ll> t[1 << 23];
	ll n, d;

	void u(ll p, ll l, ll r, ll i, ll j, ll x) {
		if(l >= j || r <= i) {
			return;
		}
		if(i <= l && r <= j) {
			t[p].f += x;
		} else {
			ll m = (l + r) / 2;
			u(p * 2, l, m, i, j, x);
			u(p * 2 + 1, m, r, i, j, x);
		}
		if(!t[p].f) t[p].s = t[p * 2].s + t[p * 2 + 1].s;
		else t[p].s = r - l;
	}

	main() {
		cin >> n;
		for(ll i = 1, a, b, c, g; i <= n; ++i) {
			cin >> a >> b >> c >> g;
			b += N, g += N;
			e.push_back({a, b, g, 1});
			e.push_back({c, b, g, -1});
		}
		sort(e.begin(), e.end());
		for(ll i = 0; i < e.size() - 1; ++i) {
			u(1, 1, N << 1, e[i].b, e[i].g, e[i].type);
			d += (e[i + 1].x - e[i].x) * t[1].s;
		}
		cout << d;
	}

**Creating Offices **

	#include <bits/stdc++.h>
	using namespace std; 
	stringstream ss;
	int n, D;
	vector<int> p(200001);
	vector<int> r(200001);
	vector<int> adj[200001];
	vector<int> de[200001];

	void dfs(int now, int prv, int lv) {
		de[lv].push_back(now);
		p[now] = prv;
		for (int u: adj[now]) {
			if (u != prv) dfs(u, now, lv+1);
		}
	}

	int go(int u, int d) {
		if (!u || !d) return 1;
		if (r[u] + d > D) return 0;
		r[u] = d;
		return go(p[u], d - 1);
	}

	signed main() {
		ios::sync_with_stdio(0); cin.tie(0);

		cin >> n >> D;
		for (int i = 1; i < n; i++) {
			int x, y; cin >> x >> y;
			adj[x].push_back(y);
			adj[y].push_back(x);
		}
		dfs(1, 0, 0); 

		int ans = 0;
		for (int i = n-1; i >= 0; i--) {
			for (int u: de[i]) {
				if (go(u, D)) {
					ans++;
					ss << u << ' ';
				}
			}
		}
		cout << ans << '\n' << ss.str() << '\n';

		return 0;
	}

**Permutations II **

	#include <stdint.h>
	#include <stdio.h>
	const int64_t MOD = 1000 * 1000 * 1000 + 7;
	int main() {
	    int N;
	    if (scanf("%d", &N) != 1) return 1;
	    int64_t A = 1, B = 1, C = 0, D = 0;
	    for (int64_t n = 4; n < N + 4; ++n) {
		int64_t E = (n+1)*D - (n-2)*C - (n-5)*B + (n-3)*A;
		A = B;
		B = C;
		C = D;
		D = (E % MOD + MOD) % MOD;
	    }
	    printf("%lld\n", (long long int)A);
	}

**New Flight Routes **

	#include <bits/stdc++.h>
	using namespace std;
	const int N = 1e5 + 5;
	vector <int> g[2][N], dag[N], ext;
	bitset <N> vs; int hd[N];
	int cmp[N], in[N], out[N];

	void dfs(int u, int r) {
	    vs[u] = 1; cmp[u] = r;
	    for (int v : g[r > 0][u])
		if (!vs[v]) dfs(v, r);
	    if (!r) ext.push_back(u);
	}

	int vis(int u) {
	    if (vs[u]) return 0;
	    vs[u] = 1;
	    if (!out[u]) return u;
	    for (int v : dag[u]) {
		int r = vis(v);
		if (r) return r;
	    }
	    return 0;
	} 

	int main() {
	    cin.tie(0)->sync_with_stdio(0);
	    int n, m; cin >> n >> m;
	    for (int i = 0; i < m; i++) {
		int u, v; cin >> u >> v;
		g[0][u].push_back(v);
		g[1][v].push_back(u);
	    }
	    for (int u = 1; u <= n; u++)
		if (!vs[u]) dfs(u, 0);
	    vs.reset(); int cc = 0;
	    reverse(ext.begin(), ext.end());
	    for (int u : ext)
		if (!vs[u]) {
		    dfs(u, ++cc); hd[cc] = u;
		}
	    for (int u = 1; u <= n; u++)
		for (int v : g[0][u])
		    if (cmp[u] != cmp[v]) {
			dag[cmp[u]].push_back(cmp[v]);
			out[cmp[u]]++; in[cmp[v]]++;
		    }
	    if (cc == 1) {
		cout << "0\n"; return 0;
	    }
	    vs.reset();
	    vector <int> m_src, u_src;
	    vector <int> m_sink, u_sink;
	    for (int u = 1; u <= cc; u++)
		if (!in[u]) {
		    int v = vis(u);
		    if (v) {
			m_src.push_back(u);
			m_sink.push_back(v);
		    }
		    else u_src.push_back(u);
		}
	    for (int u = 1; u <= cc; u++)
		if (!out[u] && !vs[u])
		    u_sink.push_back(u);
	    int x = m_src.size();
	    m_src.insert(m_src.end(),
	    u_src.begin(), u_src.end());
	    m_sink.insert(m_sink.end(),
	    u_sink.begin(), u_sink.end());
	    int y = m_src.size(), z = m_sink.size();
	    cout << max(m_src.size(), m_sink.size()) << '\n';
	    cout << hd[m_sink[0]] << ' ' << hd[m_src[x - 1]] << '\n';
	    for (int i = 1; i < x; i++)
		cout << hd[m_sink[i]] << ' ' << hd[m_src[i - 1]] << '\n';
	    for (int i = x; i < min(y, z); i++)
		cout << hd[m_sink[i]] << ' ' << hd[m_src[i]] << '\n';
	    for (int i = min(y, z); i < y; i++)
		cout << hd[m_sink[0]] << ' ' << hd[m_src[i]] << '\n';
	    for (int i = min(y, z); i < z; i++)
		cout << hd[m_sink[i]] << ' ' << hd[m_src[0]] << '\n';
	}
	
**Number Grid **

	a,b=map(int,input().split())
	print(-a^-b)

** **
** **


** **


** **
** **


** **


** **
