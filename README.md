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
